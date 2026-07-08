"""Unit tests for the model-pack certified-editing engine (hlm5/certify.py).

Deterministic, CPU-only, no model download: a hand-built d=8 / |V|=20 head
with known geometry exercises every branch of the certificate, dosing,
synthesis, whitening, and the composed admission pipeline.
"""
import pytest

torch = pytest.importorskip("torch")

from hlm5.certify import (  # noqa: E402
    ADMIT,
    ADMIT_RESCUED,
    EPS,
    REFUSE,
    admission,
    calibrate_whitening,
    certify,
    dose,
    synthesize,
    tensor_hash,
    unit,
    whiten_key,
)

D, VOCAB = 8, 20

# Named rows of the constructed head.
TARGET_A = 0        # reachable case: finite 0 < L < U
BLOCKER_LOW = 1     # competitor ahead of A with positive slope -> sets L
BLOCKER_HIGH = 2    # competitor aligned with W_A, larger norm -> sets finite U
TARGET_B = 3        # hard-blocked case
HARD_BLOCKER = 4    # a_j <= 0 and b_j == 0 against TARGET_B
TARGET_C = 5        # already-winning case: L = 0
TARGET_DUP = 6      # duplicate-row pair: synthesis cannot rescue -> REFUSE
DUP_TWIN = 7


def build_head() -> torch.Tensor:
    e = torch.eye(D)
    W = torch.zeros(VOCAB, D)
    W[TARGET_A] = e[0]
    W[BLOCKER_LOW] = e[1]
    W[BLOCKER_HIGH] = 1.5 * e[0] - e[2]
    W[TARGET_B] = e[3]
    W[HARD_BLOCKER] = e[3] + 2.0 * e[4]
    W[TARGET_C] = e[5]
    W[TARGET_DUP] = e[6]
    W[DUP_TWIN] = e[6]
    for j in range(8, VOCAB):
        W[j] = 0.01 * e[j % D]
    return W


def h_reachable() -> torch.Tensor:
    # base logits: target_A 0.1, blocker_low 1.0, blocker_high 0.15-1.0=-0.85
    h = torch.zeros(D)
    h[0], h[1], h[2] = 0.1, 1.0, 1.0
    return h


def h_hard_blocked() -> torch.Tensor:
    # base logits: target_B 0.0, hard_blocker 2.0 (a=-2, b=0 -> hard)
    h = torch.zeros(D)
    h[4], h[1] = 1.0, 0.5
    return h


def h_winning() -> torch.Tensor:
    # base argmax is already TARGET_C
    h = torch.zeros(D)
    h[5], h[1] = 2.0, 0.5
    return h


def h_duplicate() -> torch.Tensor:
    h = torch.zeros(D)
    h[6], h[1] = 1.0, 0.5
    return h


# --------------------------------------------------------------------------- #
# (a) reachable target
# --------------------------------------------------------------------------- #

def test_certificate_reachable_interval_and_blocker():
    W, h = build_head(), h_reachable()
    cert = certify(W, h, TARGET_A)

    assert cert.reachable is True
    assert cert.hard_blocker is False
    assert cert.risk in ("safe", "narrow", "brittle")
    assert cert.L == pytest.approx(0.9, abs=1e-5)       # -a/b for blocker_low
    assert cert.U == pytest.approx(1.9, abs=1e-5)       # -a/b for blocker_high
    assert cert.slack == pytest.approx(1.0, abs=1e-5)
    # candidate dose 1.05*L+1 = 1.945 >= U -> pulled back to the midpoint 1.4,
    # where the binding competitor is the upper-bound blocker
    assert cert.beta_candidate == pytest.approx(1.4, abs=1e-5)
    assert cert.blocker_id == BLOCKER_HIGH
    assert cert.margin_at_candidate == pytest.approx(0.25, abs=1e-5)
    assert cert.risk == "brittle"                        # slack 1.0 <= beta 1.4


def test_dose_lands_inside_interval_and_flips_argmax():
    W, h = build_head(), h_reachable()
    cert = certify(W, h, TARGET_A)
    beta_star = dose(cert)

    assert cert.L < beta_star < cert.U
    # analytic optimum: margins m_low = beta-0.9 and m_high = 0.95-0.5*beta
    # cross at beta = 37/30 ~ 1.2333 with worst margin ~ 1/3
    assert beta_star == pytest.approx(37.0 / 30.0, abs=0.02)
    assert cert.margin_at(beta_star) > 0.3

    # base argmax is NOT the target; dosed argmax flips to it
    assert int((W @ h).argmax()) != TARGET_A
    dosed = W @ (h + beta_star * unit(W[TARGET_A], 0))
    assert int(dosed.argmax()) == TARGET_A


def test_dose_rejects_unreachable_certificate():
    W = build_head()
    cert = certify(W, h_hard_blocked(), TARGET_B)
    with pytest.raises(ValueError):
        dose(cert)


# --------------------------------------------------------------------------- #
# (b) hard-blocked target -> synthesis rescue
# --------------------------------------------------------------------------- #

def test_certificate_hard_blocker_detected():
    W, h = build_head(), h_hard_blocked()
    cert = certify(W, h, TARGET_B)

    assert cert.reachable is False
    assert cert.hard_blocker is True
    assert cert.risk == "unreachable"
    assert cert.blocker_id == HARD_BLOCKER
    assert cert.blocker_a == pytest.approx(-2.0, abs=1e-5)   # already ahead
    assert cert.blocker_b == pytest.approx(0.0, abs=EPS)     # never overtaken


def test_synthesis_rescues_hard_blocked_target():
    W, h = build_head(), h_hard_blocked()
    r_star, m_star = synthesize(W, TARGET_B)

    assert m_star > 0.1                                   # feasible direction found
    assert r_star.norm().item() == pytest.approx(1.0, abs=1e-5)

    cert_syn = certify(W, h, TARGET_B, v=r_star)
    assert cert_syn.reachable is True
    assert cert_syn.hard_blocker is False

    beta_star = dose(cert_syn)
    dosed = W @ (h + beta_star * unit(r_star, 0))
    assert int(dosed.argmax()) == TARGET_B                # synthesized dose flips


def test_synthesis_is_deterministic():
    W = build_head()
    r1, m1 = synthesize(W, TARGET_B)
    r2, m2 = synthesize(W, TARGET_B)
    assert m1 == m2
    assert torch.equal(r1, r2)


# --------------------------------------------------------------------------- #
# (c) already-winning target -> L = 0 semantics
# --------------------------------------------------------------------------- #

def test_certificate_already_winning_has_zero_lower_bound():
    W, h = build_head(), h_winning()
    assert int((W @ h).argmax()) == TARGET_C
    cert = certify(W, h, TARGET_C)

    assert cert.reachable is True
    assert cert.L == 0.0
    assert cert.U == float("inf")
    assert cert.slack == float("inf")
    assert cert.beta_candidate == pytest.approx(1.0)      # 1.05*0 + 1
    assert cert.risk == "safe"
    assert cert.margin_at_candidate > 0                   # already positive at beta=0
    assert cert.margin_at(0.0) > 0


# --------------------------------------------------------------------------- #
# (d) whitening calibration
# --------------------------------------------------------------------------- #

def test_whitening_shapes_symmetry_and_identity_covariance():
    torch.manual_seed(0)
    hiddens = torch.randn(64, D) * 2.0 + 0.5
    mu, A = calibrate_whitening(hiddens)

    assert mu.shape == (D,)
    assert A.shape == (D, D)
    assert torch.allclose(A, A.T, atol=1e-5)              # ZCA is symmetric

    z = whiten_key(hiddens, mu, A)
    cov = (z.T @ z) / z.shape[0]
    assert (cov - torch.eye(D)).abs().max().item() < 0.05

    # whitened self-similarity: the degree-5 gate on the key's own whitened
    # vector scores exactly 1
    q = unit(whiten_key(hiddens[0], mu, A), 0)
    assert float(torch.dot(q, q) ** 5) == pytest.approx(1.0, abs=1e-5)


def test_whitening_two_pool_mu_is_key_mean_only():
    torch.manual_seed(1)
    keys = torch.randn(10, D) + 3.0
    neutrals = torch.randn(30, D) - 3.0
    mu, A = calibrate_whitening(keys, neutrals)
    assert torch.allclose(mu, keys.mean(0))               # mu excludes neutral pool
    assert A.shape == (D, D)


# --------------------------------------------------------------------------- #
# (e) admission end-to-end
# --------------------------------------------------------------------------- #

def test_admission_admits_reachable_target():
    W, h = build_head(), h_reachable()
    res = admission(W, h, TARGET_A)

    assert res.decision == ADMIT
    assert res.rescued is False
    assert res.certificate.L < res.dose < res.certificate.U
    assert res.value.norm().item() == pytest.approx(res.dose, abs=1e-4)
    assert int((W @ (h + res.value)).argmax()) == TARGET_A

    rec = res.receipt
    assert rec["decision"] == ADMIT
    assert rec["risk"] == "brittle"
    assert rec["L"] == pytest.approx(0.9, abs=1e-3)
    assert rec["U"] == pytest.approx(1.9, abs=1e-3)
    assert rec["blocker_id"] == BLOCKER_HIGH
    assert rec["beta_star"] == pytest.approx(res.dose, abs=1e-3)
    assert rec["margin_at_beta_star"] > 0
    assert rec["synthesis_min_margin"] is None
    assert len(rec["key_hash"]) == 16
    assert rec["value_hash"] == tensor_hash(res.value)


def test_admission_rescues_hard_blocked_target():
    W, h = build_head(), h_hard_blocked()
    res = admission(W, h, TARGET_B)

    assert res.decision == ADMIT_RESCUED
    assert res.rescued is True
    assert res.certificate_naive.reachable is False
    assert res.certificate_naive.hard_blocker is True
    assert res.certificate.reachable is True
    assert res.synthesis_min_margin > 0
    assert int((W @ (h + res.value)).argmax()) == TARGET_B   # stored value flips

    rec = res.receipt
    assert rec["decision"] == ADMIT_RESCUED
    assert rec["rescued"] is True
    assert rec["hard_blocker"] is True                       # naive cert was blocked
    assert rec["synthesis_min_margin"] > 0
    assert rec["blocker_a"] == pytest.approx(-2.0, abs=1e-3)


def test_admission_admits_already_winning_target_with_zero_L():
    W, h = build_head(), h_winning()
    res = admission(W, h, TARGET_C)

    assert res.decision == ADMIT
    assert res.certificate.L == 0.0
    assert res.receipt["L"] == 0.0
    assert res.receipt["U"] == "inf"                          # receipt inf convention
    assert res.receipt["risk"] == "safe"
    assert int((W @ (h + res.value)).argmax()) == TARGET_C


def test_admission_refuses_when_synthesis_cannot_rescue():
    W, h = build_head(), h_duplicate()
    res = admission(W, h, TARGET_DUP)

    assert res.decision == REFUSE
    assert res.dose is None
    assert res.value is None
    assert res.synthesis_min_margin is not None
    assert res.synthesis_min_margin <= 0                     # duplicate row: m* <= 0
    assert res.certificate.hard_blocker is True
    assert res.certificate.blocker_id == DUP_TWIN

    rec = res.receipt
    assert rec["decision"] == REFUSE
    assert rec["risk"] == "unreachable"
    assert rec["beta_star"] is None
    assert rec["value_hash"] is None
    assert len(rec["key_hash"]) == 16


def test_admission_whitened_key_hash_matches_deployed_gate_key():
    torch.manual_seed(2)
    W, h = build_head(), h_reachable()
    hiddens = torch.stack([h_reachable(), h_winning(), h_hard_blocked()])
    neutrals = torch.randn(16, D)
    mu, A = calibrate_whitening(hiddens, neutrals)

    res_plain = admission(W, h, TARGET_A)
    res_white = admission(W, h, TARGET_A, whiten=(mu, A))

    assert res_white.receipt["key_hash"] == tensor_hash(unit(whiten_key(h, mu, A), 0))
    assert res_white.receipt["key_hash"] != res_plain.receipt["key_hash"]
    # whitening changes only the receipt's key identity, not the certificate
    assert res_white.receipt["L"] == res_plain.receipt["L"]
    assert res_white.receipt["beta_star"] == res_plain.receipt["beta_star"]
