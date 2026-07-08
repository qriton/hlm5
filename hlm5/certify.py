"""Certified-editing engine for the HLM5 model pack.

Port of the validated HLM5 certificate machinery (D:\\HLM5, validated
2026-07-02/03) into a dependency-light module: pure torch + stdlib, no hlm_kb
imports. Sources ported (conventions followed exactly -- EPS values, risk-label
thresholds, open feasible interval, dose cap):

  - baselines/run_1b_certificate.py       certificate envelope + risk labels
  - baselines/run_1b_betastar.py          beta* grid dose on the capped interval
  - baselines/run_1b_faithful_certdosed.py  cert-dosed convention (store value at
                                          norm beta*, attach with boost 1.0)
  - baselines/run_1b_synth_verify.py      synthesis flip verification
  - demos/read_and_memorize.py            admission pipeline + receipts
  - baselines/run_1b_faithful.py          ZCA whitening calibration

The certificate is EXACT for a tied, bias-free linear head: the post-edit
margin of target y over competitor j at edit strength beta is affine,

    m_j(beta) = a_j + beta * b_j,   a_j = (W_y - W_j) . h,   b_j = (W_y - W_j) . v

so the feasible interval is

    L = max(0, max_{b_j > EPS} -a_j / b_j),    U = min_{b_j < -EPS} -a_j / b_j

with the target reachable iff L < U and no competitor is a HARD BLOCKER
(a_j <= 0 AND b_j <= EPS: already ahead and never overtaken along v).

Deployed gate conventions (run_1b_faithful.py): whitened keys, degree-5 kernel,
TEMPERATURE = 0.10, hard gate GATE_THRESH = 0.95.

This is the reference implementation of the paper's Section on reachability
certificates.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

import torch

# Deployed constants (baselines/run_1b_faithful.py, run_1b_faithful_certdosed.py)
EPS = 1e-4            # slope dead-zone for the certificate envelope
GATE_THRESH = 0.95    # hard gate tau on the degree-5 kernel score
TEMPERATURE = 0.10
DEGREE = 5

ADMIT = "ADMIT"
ADMIT_RESCUED = "ADMIT_RESCUED"
REFUSE = "REFUSE"


def unit(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def tensor_hash(t: torch.Tensor) -> str:
    """16-hex-char sha256 of the fp32 bytes (demos/read_and_memorize.py sha16)."""
    flat = t.detach().to("cpu", torch.float32).contiguous().reshape(-1)
    return hashlib.sha256(struct.pack(f"{flat.numel()}f", *flat.tolist())).hexdigest()[:16]


@dataclass
class Certificate:
    """Reachability certificate for one (hidden h, target y, value v) triple.

    beta_candidate is the demo dose formula (demos/read_and_memorize.py):
    1.05*L + 1.0, pulled back to the interval midpoint when it would land at or
    above U. margin_at_candidate is the worst-case competitor margin at that
    dose; blocker_id is the binding competitor there (or, when hard-blocked,
    the worst hard blocker by base logit).
    """
    target_id: int
    L: float
    U: float                       # float("inf") when no negative slope binds
    slack: float                   # U - L (inf when U is inf)
    reachable: bool
    hard_blocker: bool
    blocker_id: int
    risk: str                      # safe | narrow | brittle | unreachable
    beta_candidate: float
    margin_at_candidate: float
    blocker_a: float | None = None  # populated only for hard blockers
    blocker_b: float | None = None
    v: torch.Tensor = field(default=None, repr=False)              # value direction certified
    aj: torch.Tensor = field(default=None, repr=False)             # (V-1,) competitor intercepts
    bj: torch.Tensor = field(default=None, repr=False)             # (V-1,) competitor slopes
    competitor_ids: torch.Tensor = field(default=None, repr=False)  # (V-1,) vocab ids

    def margin_at(self, beta: float) -> float:
        """Worst-case competitor margin min_j (a_j + beta*b_j) at dose beta."""
        return float((self.aj + beta * self.bj).min())


@dataclass
class AdmissionResult:
    """Outcome of the composed admission pipeline (demos/read_and_memorize.py)."""
    decision: str                          # ADMIT | ADMIT_RESCUED | REFUSE
    rescued: bool
    certificate: Certificate               # governing certificate (synth cert if rescued)
    certificate_naive: Certificate         # certificate under v = unit(W_y)
    dose: float | None                     # beta* (None on refusal)
    value: torch.Tensor | None             # unit(v) * beta*, ready to store (attach boost=1.0)
    synthesis_min_margin: float | None     # m* when synthesis was attempted
    receipt: dict                          # JSON-ready receipt payload


@torch.no_grad()
def certify(W: torch.Tensor, h: torch.Tensor, target_id: int,
            v: torch.Tensor | None = None, *, eps: float = EPS) -> Certificate:
    """Certificate envelope. Ports envelope() from baselines/run_1b_certificate.py
    / run_1b_faithful_certdosed.py / demos/read_and_memorize.py.

    W: (V, d) head weight (tied, bias-free -> the envelope is exact).
    h: (d,) hidden state at the key's last position.
    v: value direction; defaults to the deployed naive value unit(W_y).
    """
    W = W.detach().float()
    h = h.detach().float().reshape(-1)
    V = W.shape[0]
    device = W.device
    base = W @ h
    if v is None:
        v = W[target_id] / (W[target_id].norm() + 1e-8)
    v = v.detach().float().reshape(-1)
    a = base[target_id] - base                       # a_j (a[target_id] = 0)
    Wv = W @ v
    b = Wv[target_id] - Wv                           # b_j (b[target_id] = 0)
    mask = torch.ones(V, dtype=torch.bool, device=device)
    mask[target_id] = False
    aj, bj = a[mask], b[mask]
    idx = torch.arange(V, device=device)[mask]
    bpos, bneg, bzero = bj > eps, bj < -eps, bj.abs() <= eps
    # hard blocker: competitor already ahead (a_j <= 0) that is never overtaken
    # along v (b_j <= eps)
    hard = (bzero | (bj < 0)) & (aj <= 0)
    ratio = -aj / bj
    L = float(torch.clamp(ratio[bpos].max(), min=0.0)) if bool(bpos.any()) else 0.0
    U = float(ratio[bneg].min()) if bool(bneg.any()) else float("inf")
    reachable = (not bool(hard.any())) and L < U

    # candidate dose (demo formula): beta = 1.05*L + 1, kept inside the envelope
    beta = 1.05 * L + 1.0
    if math.isfinite(U) and beta >= U:
        beta = 0.5 * (L + U)
    marg = aj + beta * bj
    margin_at_candidate = float(marg.min())
    blocker_id = int(idx[marg.argmin()])             # binding competitor at beta
    blocker_a = blocker_b = None
    if reachable:
        s = U - L
        risk = "safe" if s > 5 * beta else ("narrow" if s > beta else "brittle")
    else:
        risk = "unreachable"
        if bool(hard.any()):
            hard_idx = idx[hard]
            jb = hard_idx[base[hard_idx].argmax()]   # worst hard blocker by base logit
            blocker_id = int(jb)
            blocker_a = float(a[jb])
            blocker_b = float(b[jb])
    return Certificate(
        target_id=int(target_id), L=L, U=U,
        slack=(U - L) if math.isfinite(U) else float("inf"),
        reachable=reachable, hard_blocker=bool(hard.any()), blocker_id=blocker_id,
        risk=risk, beta_candidate=beta, margin_at_candidate=margin_at_candidate,
        blocker_a=blocker_a, blocker_b=blocker_b,
        v=v, aj=aj, bj=bj, competitor_ids=idx)


@torch.no_grad()
def dose(cert: Certificate, *, grid: int = 120, cap: float | None = None) -> float:
    """beta* = argmax_{beta in [L, min(U, cap)]} min_j (a_j + beta*b_j).

    Ports baselines/run_1b_betastar.py: the worst-case margin is 1-D concave
    piecewise-linear in beta, solved by scanning a dense grid on the capped
    interval. Default cap is the reference convention L + max(50, 5*L).
    """
    if not cert.reachable:
        raise ValueError("dose() requires a reachable certificate")
    Lf, Uf = cert.L, cert.U
    if cap is None:
        cap = Lf + max(50.0, 5.0 * Lf)
    hi = min(Uf, cap) if math.isfinite(Uf) else cap
    if hi <= Lf:
        raise ValueError(f"empty dosing interval: L={Lf}, min(U, cap)={hi}")
    device = cert.aj.device
    betas = Lf + torch.linspace(0.0, 1.0, grid, device=device) * (hi - Lf)
    marg = cert.aj[:, None] + betas[None, :] * cert.bj[:, None]   # (V-1, grid)
    worst = marg.min(0).values                                    # (grid,)
    return float(betas[int(worst.argmax())])


@torch.no_grad()
def synthesize(W: torch.Tensor, target_id: int, *, steps: int = 300,
               lr: float = 0.5) -> tuple[torch.Tensor, float]:
    """r* = argmax_{||r|| <= 1} min_{j != y} (W_y - W_j) . r  and its margin m*.

    Ports synthesize() from run_1b_faithful_certdosed.py / demos/
    read_and_memorize.py (verified end-to-end in run_1b_synth_verify.py):
    projected gradient ascent on a soft-min of the competitor margins, with the
    softmin sharpness annealed 8.0 -> 40.0 (x1.01/step). m* > 0 certifies that
    every competitor slope is positive along r*, i.e. the target is reachable.
    """
    W = W.detach().float()
    r = (W[target_id] / (W[target_id].norm() + 1e-8)).clone()
    tau = 8.0
    for _ in range(steps):
        sc = W @ r
        sc_comp = sc.clone()
        sc_comp[target_id] = -1e9
        w = torch.softmax(tau * (sc_comp - sc_comp.max()), dim=0)
        grad = W[target_id] - (w[:, None] * W).sum(0)
        r = r + lr * grad
        r = r / (r.norm() + 1e-8)
        tau = min(40.0, tau * 1.01)
    sc = W @ r
    sc_comp = sc.clone()
    sc_comp[target_id] = -1e9
    m_star = float(sc[target_id] - sc_comp.max())    # true min-margin to nearest competitor
    return r, m_star


@torch.no_grad()
def calibrate_whitening(hidden_states: torch.Tensor,
                        neutral_states: torch.Tensor | None = None, *,
                        floor_frac: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """ZCA whitening (mu, A) from key hiddens (+ optional neutral-text hiddens).

    Ports the calibration recipe from baselines/run_1b_faithful.py / demos/
    read_and_memorize.py: mu is the mean of the KEY hiddens only; the covariance
    pools key + neutral residuals about mu; eigenvalues are floored at
    floor_frac * mean(eig) before the inverse square root. A is symmetric (ZCA).
    Whitened key = (h - mu) @ A.
    """
    H = hidden_states.detach().float()
    if H.ndim != 2:
        raise ValueError("hidden_states must have shape (N, d)")
    mu = H.mean(0)
    pool = H if neutral_states is None else torch.cat([H, neutral_states.detach().float()])
    resid = pool - mu
    cov = (resid.T @ resid) / resid.shape[0]
    eigval, eigvec = torch.linalg.eigh(cov)
    floor = floor_frac * float(eigval.mean())
    A = eigvec @ torch.diag((eigval + floor).rsqrt()) @ eigvec.T
    return mu, A


def whiten_key(h: torch.Tensor, mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Whitened gate key (h - mu) @ A, matching the deployed key_transform."""
    return (h - mu) @ A


def _round_or_inf(x: float, nd: int = 3):
    return round(x, nd) if math.isfinite(x) else "inf"


@torch.no_grad()
def admission(W: torch.Tensor, h: torch.Tensor, target_id: int, *,
              whiten: tuple[torch.Tensor, torch.Tensor] | None = None,
              eps: float = EPS, grid: int = 120, cap: float | None = None,
              synth_steps: int = 300, synth_lr: float = 0.5) -> AdmissionResult:
    """Composed admission pipeline (demos/read_and_memorize.py steps d-e).

    1. Certificate under the deployed naive value v = unit(W_y).
    2. If unreachable, residual-synthesis rescue: r* with m* > 0 re-certifies.
    3. If still unreachable, REFUSE with the blocker in the receipt.
    4. Otherwise dose at beta* (grid argmax of the worst-case margin,
       run_1b_betastar.py) and return the cert-dosed value unit(v) * beta*,
       to be stored at that norm and attached with boost 1.0
       (run_1b_faithful_certdosed.py convention).

    whiten: optional (mu, A) so the receipt's key_hash matches the deployed
    whitened gate key; otherwise the raw hidden is hashed.
    """
    cert_naive = certify(W, h, target_id, eps=eps)
    cert = cert_naive
    rescued = False
    m_star: float | None = None
    if not cert_naive.reachable:
        r_star, m_star = synthesize(W, target_id, steps=synth_steps, lr=synth_lr)
        if m_star > 0:
            cert_syn = certify(W, h, target_id, v=r_star, eps=eps)
            if cert_syn.reachable:
                cert, rescued = cert_syn, True

    key_vec = whiten_key(h.detach().float(), whiten[0], whiten[1]) if whiten is not None \
        else h.detach().float()
    key_hash = tensor_hash(unit(key_vec, 0))

    receipt = {
        "target_id": int(target_id),
        "reachable": cert.reachable,
        "rescued": rescued,
        "hard_blocker": cert_naive.hard_blocker,
        "L": _round_or_inf(cert.L),
        "U": _round_or_inf(cert.U),
        "slack": _round_or_inf(cert.slack),
        "risk": cert.risk,
        "blocker_id": cert.blocker_id,
        "beta_candidate": round(cert.beta_candidate, 3),
        "synthesis_min_margin": round(m_star, 4) if m_star is not None else None,
        "key_hash": key_hash,
        "eps": eps,
    }
    if cert_naive.blocker_a is not None:
        receipt["blocker_a"] = round(cert_naive.blocker_a, 4)
        receipt["blocker_b"] = round(cert_naive.blocker_b, 4)

    if not cert.reachable:
        receipt.update(decision=REFUSE, beta_star=None, value_hash=None,
                       margin_at_beta_star=None)
        return AdmissionResult(decision=REFUSE, rescued=False, certificate=cert,
                               certificate_naive=cert_naive, dose=None, value=None,
                               synthesis_min_margin=m_star, receipt=receipt)

    beta_star = dose(cert, grid=grid, cap=cap)
    value = unit(cert.v, 0) * beta_star              # store at norm beta*, attach boost=1.0
    decision = ADMIT_RESCUED if rescued else ADMIT
    receipt.update(decision=decision, beta_star=round(beta_star, 3),
                   margin_at_beta_star=round(cert.margin_at(beta_star), 4),
                   value_hash=tensor_hash(value))
    return AdmissionResult(decision=decision, rescued=rescued, certificate=cert,
                           certificate_naive=cert_naive, dose=beta_star, value=value,
                           synthesis_min_margin=m_star, receipt=receipt)
