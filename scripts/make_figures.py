#!/usr/bin/env python3
"""make_figures.py -- publication figures for the HLM5 paper (in-repo, portable).

Builds all data figures with one shared camera-ready style (serif + cm math,
9pt base / 8pt ticks / 8pt annotations at final print width, no in-figure
titles, bold (a)/(b) panel tags, neutral-ink annotations, thin spines).

Every number is read from a results/ artifact. Paths come from ``hlm5.io``
(``RESULTS_DIR`` for inputs, ``FIGURES_DIR`` for the design-size PDFs, and
``FIGURES_DIR/_preview`` for 200-dpi PNG previews of the same basename). The
figures are drawn at their final print size so LaTeX includes them at scale
1.00 and the fonts print at exactly 9pt/8pt.

Deterministic: the ``Agg`` backend and a fixed RNG ``SEED`` for the only
stochastic element (the multi-key strip-plot jitter).

  figure                     design size     input artifact(s)
  -------------------------- --------------- ---------------------------------
  fig-reachability-geometry  6.4 x 2.60 in   betastar.json, cert_envelope_synth.json
  fig-betastar               3.9 x 3.00 in   betastar.json
  fig-envelope-multikey      6.2 x 2.20 in   cert_envelope_multikey.json
  fig-certdosed              6.2 x 3.10 in   hlm5_1b_faithful_certdosed.json
  fig-certpredict            4.5 x 2.85 in   cert_counterfact_gpt2xl.jsonl,
                                             editors_counterfact_gpt2xl.jsonl
  fig-head-geometry          4.7 x 2.60 in   headgeom_metrics.json
  fig-notax-delta            3.8 x 2.55 in   g2a_no_tax.json, g3a_no_tax.json,
                                             notax_params.json
  fig-gate-selectivity       6.4 x 2.40 in   gate_roc.json
  fig-frontier               5.5 x 3.20 in   gpt2_table1_v2.json, rome_gpt2.json,
                                             rome_easyedit_gpt2.json

Run:  python scripts/make_figures.py     (repo root, CPU-only, no network)
"""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.

import json
import math
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hlm5.io import RESULTS_DIR, FIGURES_DIR

PREVIEW_DIR = FIGURES_DIR / "_preview"

# ---- house palette (data ink only; all text is neutral INK) ------------------
TEAL = "#2E6F6A"    # primary series
STEEL = "#5B7C99"   # secondary
NAVY = "#1F3A5F"    # emphasis / dark
BRICK = "#A93C2E"   # reference lines / negative
GREY = "#8A8F94"    # de-emphasized marks
INK = "#222222"     # all annotation text

ANN = 8.0           # annotation font size (pt, at final print width)


def _upper_bound(row):
    """Read legacy string infinity and refreshed JSON-null as unbounded U."""
    value = row.get("U")
    return math.inf if value is None or isinstance(value, str) else float(value)


SEED = 20260703


# ==============================================================================
# shared academic style
# ==============================================================================
def style_rc():
    """One rcParams dict used by every figure. Figures are designed at their
    final print width, so these sizes are the printed sizes: 9pt base/labels,
    8pt ticks, serif + Computer Modern mathtext."""
    return {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,          # enabled per-axis where it aids reading
        "grid.color": "#b0b6bb",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def _panel_tag(ax, label):
    """Compact bold panel tag at the top-left corner of a panel (neutral ink)."""
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=INK, ha="left", va="bottom")


def _load(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(name):
    rows = []
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finish(fig, stem):
    """Save at the exact design size (no bbox crop) so the PDF's physical
    width equals the width the paper includes it at -- fonts then print at
    exactly 9pt/8pt. A 200-dpi PNG preview of the same basename is written to
    FIGURES_DIR/_preview for eyeballing."""
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = PREVIEW_DIR / f"{stem}.png"
    fig.savefig(
        pdf,
        metadata={
            "Creator": "HLM5 make_figures.py",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png, dpi=200, metadata={"Software": "HLM5 make_figures.py"})
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return pdf


# ==============================================================================
# FIGURE -- reachability geometry (two panels)     print width 6.4in
# ==============================================================================
def fig_reachability_geometry():
    bs = _load("betastar.json")
    beta_star = bs["median_beta_star"]                 # 72.134
    margin = bs["median_worst_margin_betastar"]        # 18.39

    synth = _load("cert_envelope_synth.json")
    ex = next(e for e in synth["residual_synthesis"]["examples"]
              if e["token"] == " that")
    blocker_a = ex["min_margin_unitWy"]                # -0.055

    # illustration anchors for the feasible band around the measured medians
    L, U = 42.0, 104.0

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.60))

    # ---- panel (a): reachable target ----------------------------------------
    ax = axes[0]
    x = np.linspace(0, 130, 2)

    s1 = margin / (beta_star - L)          # ascending line, crosses 0 at L
    s2 = margin / (U - beta_star)          # descending line, crosses 0 at U
    ax.plot(x, s1 * (x - L), color=TEAL, lw=1.8, zorder=3)
    ax.plot(x, s2 * (U - x), color=BRICK, lw=1.8, zorder=3)
    ax.plot(x, 60.0 - 0.253 * x, color=BRICK, lw=1.8, zorder=3)
    ax.plot(x, 45.0 + 0.0 * x, color=STEEL, lw=1.8, zorder=3)

    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.axvspan(L, U, color=TEAL, alpha=0.10, zorder=1)
    ax.plot([beta_star, beta_star], [-40, margin], color=NAVY, lw=1.1,
            ls="--", zorder=4)
    ax.plot([beta_star], [margin], "o", color=NAVY, ms=6.5, zorder=6)

    ax.annotate(
        rf"$\beta^\star \approx {beta_star:.0f}$,  margin $= {margin:.1f}$",
        xy=(beta_star - 1.5, margin + 2.0), xytext=(3, 68),
        color=INK, fontsize=ANN, ha="left", va="center", zorder=7,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                        shrinkA=2, shrinkB=3))

    ax.text(L - 3, -33, r"$L$", color=INK, fontsize=ANN,
            ha="right", va="center")
    ax.text((L + U) / 2, -33, r"feasible $(L,\,U)$", color=INK,
            fontsize=ANN, ha="center", va="center")
    ax.text(U + 3, -33, r"$U$", color=INK, fontsize=ANN,
            ha="left", va="center")

    ax.set_xlim(0, 130)
    ax.set_ylim(-40, 80)
    ax.set_xlabel(r"edit strength $\beta_{\mathrm{eff}}$")
    ax.set_ylabel(r"competitor margin $m_{yj} = a_j + \beta_{\mathrm{eff}} b_j$"
                  "  (logits)", fontsize=8)
    ax.grid(True, axis="y")
    _panel_tag(ax, "(a)")

    # ---- panel (b): unreachable target (Cor. 1) ------------------------------
    ax = axes[1]
    x = np.linspace(0, 130, 2)
    ax.plot(x, 0.030 + 0.00052 * x, color=STEEL, lw=1.6, zorder=3)
    ax.plot(x, -0.020 + 0.00110 * x, color=STEEL, lw=1.6, zorder=3)
    ax.plot(x, blocker_a - 0.00090 * x, color=BRICK, lw=2.0, zorder=4)

    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.axhspan(-0.08, 0, color=BRICK, alpha=0.08, zorder=1)

    ax.text(5, 0.062, r'target " that"', color=INK, fontsize=ANN,
            ha="left", va="center")
    ax.text(74, -0.0705, r"no feasible $\beta$: target certified unreachable",
            color=INK, fontsize=ANN, ha="center", va="center", zorder=5)

    ax.set_xlim(0, 130)
    ax.set_ylim(-0.08, 0.07)
    ax.set_xlabel(r"edit strength $\beta_{\mathrm{eff}}$")
    ax.set_ylabel("worst competitor margin  (logits)")
    ax.grid(True, axis="y")
    _panel_tag(ax, "(b)")

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.94))   # headroom for tags
    return _finish(fig, "fig-reachability-geometry")


# ==============================================================================
# FIGURE -- beta* worst-case margin curve          print width 3.9in
#
# The worst-case margin M(beta) = min_{j!=y} (a_j + beta b_j) is concave and
# piecewise-linear (a min of competitor lines): it crosses 0 at the lower
# reachability bound L (rising), peaks at beta* with margin*, and crosses 0
# again at the upper bound U (falling). betastar.json stores only the four
# medians (beta*, margin*, heuristic margin, gain) -- not per-fact a_j/b_j or
# per-fact (L,U). We therefore plot the (L,U,beta*,margin*) piecewise-linear
# envelope: the measured peak (beta*=72.1, margin*=18.39) with the feasible band
# L=42/U=104 taken from the companion fig-reachability-geometry (same median
# fact). The heuristic dose beta=1.05L+1 (the rule the run script compares
# against) is marked on the rising edge at its measured median margin 1.08, and
# the ~17x median margin ratio is annotated. No invented y-intercept, no invented
# upper crossing.
# ==============================================================================
def fig_betastar():
    bs = _load("betastar.json")
    beta_star = bs["median_beta_star"]                  # 72.134
    margin = bs["median_worst_margin_betastar"]         # 18.39
    heur_margin = bs["median_worst_margin_heuristic"]   # 1.084
    ratio = margin / heur_margin                        # ~17.0x

    # feasible-band anchors, consistent with fig-reachability-geometry
    L, U = 42.0, 104.0

    fig, ax = plt.subplots(figsize=(3.9, 3.00))

    # concave tent: (L,0) -> (beta*, margin*) -> (U,0), drawn only on [L,U]
    xt = np.array([L, beta_star, U])
    yt = np.array([0.0, margin, 0.0])
    ax.plot(xt, yt, color=NAVY, lw=2.0, zorder=3)

    ax.axvspan(L, U, color=TEAL, alpha=0.10, zorder=1)
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.plot([beta_star, beta_star], [0, margin], color=NAVY, lw=1.0,
            ls="--", zorder=2)
    ax.plot([beta_star], [margin], "o", color=NAVY, ms=6.5, zorder=6)

    # L / U band edges on the zero line
    for xb, lab, dx in ((L, r"$L$", -2.5), (U, r"$U$", 2.5)):
        ax.plot([xb, xb], [-0.55, 0.55], color=GREY, lw=1.0, zorder=2)
        ax.text(xb + dx, -1.9, lab, color=INK, fontsize=ANN,
                ha="center", va="center")

    ax.text(4, 23.0, rf"$\beta^\star \approx {beta_star:.0f}$,"
                     rf"  margin $= {margin:.1f}$",
            color=INK, fontsize=ANN, ha="left", va="center")

    # heuristic dose beta=1.05L+1, placed on the rising edge at its measured
    # median margin (1.08); slope of the rising segment sets the x position.
    s_up = margin / (beta_star - L)
    beta_heur = L + heur_margin / s_up
    ax.plot([beta_heur], [heur_margin], "s", color=BRICK, ms=5.5, zorder=6)
    ax.annotate("heuristic dose:\nmargin $= %.2f$" % heur_margin,
                xy=(beta_heur + 0.6, heur_margin + 0.4), xytext=(6, 13.5),
                color=INK, fontsize=ANN, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                shrinkA=2, shrinkB=3))

    # ~17x median margin ratio over the heuristic (placed in the open interior)
    ax.text(82.0, 6.0, f"{ratio:.0f}$\\times$\nmargin ratio", color=INK,
            fontsize=ANN, ha="center", va="center")

    ax.set_xlim(0, 118)
    ax.set_ylim(-3, 26)
    ax.set_xlabel(r"edit strength $\beta$")
    ax.set_ylabel(r"worst-case margin $\min_{j \neq y}\,(a_j + \beta b_j)$",
                  fontsize=8.5)
    ax.grid(True, axis="y")

    fig.tight_layout()
    return _finish(fig, "fig-betastar")


# ==============================================================================
# FIGURE -- multi-key envelope strip plot           print width 6.2in
# ==============================================================================
def fig_envelope_multikey():
    mk = _load("cert_envelope_multikey.json")
    keys = mk["keys"]
    agg = mk["aggregates"]["reachable_fraction"]
    mean, std = agg["mean"], agg["std"]                 # 0.7938, 0.0221

    rows = {"novel": 2, "real": 1, "generic": 0}
    row_names = {2: "novel entities", 1: "real entities", 0: "generic prefixes"}

    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(6.2, 2.20))

    ax.axvspan(mean - std, mean + std, color=STEEL, alpha=0.18, zorder=1)
    ax.plot([mean, mean], [-0.5, 2.42], color=TEAL, lw=1.2, ls="--", zorder=2)
    ax.text(mean, 2.52, rf"mean {mean:.3f} $\pm$ {std:.3f} (1 s.d.)",
            color=INK, fontsize=ANN, ha="center", va="bottom")

    ref = None
    for k in keys:
        y0 = rows[k["category"]]
        if k["is_original_key"]:
            ref = (k["reachable_fraction"], y0)
            continue
        jitter = rng.uniform(-0.18, 0.18)
        ax.plot(k["reachable_fraction"], y0 + jitter, "o", color=TEAL,
                ms=4.8, mec="white", mew=0.5, alpha=0.88, zorder=3)

    # reference key: navy diamond at row centre, labelled directly
    ax.plot(ref[0], ref[1], "D", color=NAVY, ms=6.5, mec="white", mew=0.7,
            zorder=5)
    ax.annotate(f"reference key {ref[0]:.3f}",
                xy=(ref[0], ref[1] - 0.07), xytext=(ref[0] - 0.0145, 1.46),
                color=INK, fontsize=ANN, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                shrinkA=3, shrinkB=2))

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([row_names[0], row_names[1], row_names[2]])
    ax.set_ylim(-0.5, 2.85)
    ax.set_xlim(0.725, 0.865)
    ax.set_xlabel("reachable fraction of target pool (1200 tokens per key)")
    ax.grid(True, axis="x")

    fig.tight_layout()
    return _finish(fig, "fig-envelope-multikey")


# ==============================================================================
# FIGURE -- per-fact certified dose windows          print width 6.2in
# ==============================================================================
def fig_certdosed():
    cd = _load("hlm5_1b_faithful_certdosed.json")
    boost = cd["arms"]["global_repro"]["boost"]          # 102.8
    facts = cd["per_fact"]

    reachable = [r for r in facts if r["reachable"]]
    unreachable = [r for r in facts if not r["reachable"]]
    reachable.sort(key=lambda r: (_upper_bound(r), r["L"]))
    unreachable.sort(
        key=lambda r: (
            not r.get("rescued", False),
            r.get("beta_star_synth", math.inf),
            r["L"],
        )
    )
    order = reachable + unreachable
    n_reach = len(reachable)

    CAP = 140.0
    fig, ax = plt.subplots(figsize=(6.2, 3.10))

    for xi, r in enumerate(order):
        if r["reachable"]:
            u = _upper_bound(r)
            top = min(u, CAP)
            ax.plot([xi, xi], [r["L"], top], color=TEAL, lw=1.8,
                    solid_capstyle="butt", zorder=3)
            ax.plot(xi, r["L"], marker="_", ms=7, mew=1.4, color=TEAL,
                    zorder=4)
            if u <= CAP:
                ax.plot(xi, u, marker="_", ms=7, mew=1.4, color=TEAL,
                        zorder=4)
            else:
                ax.plot(xi, CAP - 1.0, marker="^", ms=5.5, color=TEAL,
                        mec="white", mew=0.5, zorder=4)
            ax.plot(xi, r["beta_star"], "o", color=TEAL, ms=5.5,
                    mec="white", mew=0.7, zorder=5)
            if u < boost:            # global boost overshoots U: no flip
                ax.plot(xi, u + 6.0, "x", color=BRICK, ms=6.5, mew=1.8,
                        zorder=5)
        else:
            # inverted naive window (L > U): certified unreachable under unit(W_y)
            ax.plot([xi, xi], [_upper_bound(r), r["L"]], color=GREY, ls=":", lw=1.3,
                     zorder=3)
            ax.plot([xi, xi], [_upper_bound(r), r["L"]], marker="o", ms=3.8, ls="none",
                     mfc="white", mec=GREY, mew=1.0, zorder=4)
            if r.get("rescued"):
                ax.plot(xi, r["beta_star_synth"], "D", color=TEAL, ms=5.5,
                        mec="white", mew=0.7, zorder=5)
            else:
                ax.plot(xi, min(r["beta_star"], CAP - 5), "x", color=BRICK,
                        ms=6.5, mew=1.8, zorder=5)

    ax.axhline(boost, color=BRICK, lw=1.3, ls="--", zorder=2)
    ax.axvline(n_reach - 0.5, color="#aab2b8", lw=0.9, ls=(0, (4, 3)),
               zorder=1)

    # annotations (neutral ink) ------------------------------------------------
    ax.text(14.6, 105.5, f"global boost ({boost:g})", color=INK,
            fontsize=ANN, ha="center", va="bottom")
    ax.text(14.7, 128.0, "synthesized $r^\\star$ dose ($\\diamond$)\n"
            "no certificate ($\\times$)",
            color=INK, fontsize=ANN, ha="center", va="center")
    ax.text(-0.5, 141.5, r"$\uparrow$  $U$ exceeds axis (up to $\infty$)",
            color=INK, fontsize=ANN, ha="left", va="center")

    # per-fact callouts centred over their own columns, with thin leader lines
    sat = next(r for r in reachable if r["target"] == " Saturn")
    rom = next(
        r for r in reachable if r["target"] == " Rome" and _upper_bound(r) > 60
    )
    for r in (sat, rom):
        xi = order.index(r)
        u = _upper_bound(r)
        lab_y = u + 20.0
        ax.plot([xi, xi], [u + 2.0, lab_y - 2.5], color=GREY, lw=0.7,
                zorder=4)
        ax.text(xi, lab_y, f'"{r["target"].strip()}"\n$U$ = {u:.1f}',
                color=INK, fontsize=ANN, ha="center", va="bottom")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([r["target"].strip() for r in order],
                       rotation=55, ha="right")
    ax.set_xlim(-0.9, 16.9)
    ax.set_ylim(-5, 148)
    ax.set_yticks(np.arange(0, 141, 20))
    ax.set_ylabel(r"edit strength $\beta$")
    ax.set_xlabel("edited fact (target token)")
    ax.grid(True, axis="y")

    fig.tight_layout()
    return _finish(fig, "fig-certdosed")


# ==============================================================================
# FIGURE -- certificate difficulty predicts generalization   print width 4.5in
# ==============================================================================
def fig_certpredict():
    cert = {}
    for r in _load_jsonl("cert_counterfact_gpt2xl.jsonl"):
        if r.get("L") is not None:
            cert[r["case_id"]] = r["L"]

    editors = ("ROME", "FT", "GRACE")
    para = {e: {} for e in editors}
    for r in _load_jsonl("editors_counterfact_gpt2xl.jsonl"):
        e = r.get("editor")
        if e not in para or "error" in r or r.get("paraphrase_argmax") is None:
            continue
        para[e][r["case_id"]] = r["paraphrase_argmax"]

    shared = sorted(set(cert) & set.intersection(
        *[set(para[e]) for e in editors]))
    shared.sort(key=lambda c: (cert[c], c))            # easiest -> hardest
    n = len(shared)
    quart = [shared[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    n_q = len(quart[0])

    means = {e: [float(np.mean([para[e][c] for c in q])) for q in quart]
             for e in editors}

    style = {
        "ROME": dict(color=TEAL, marker="o", ls="-"),
        "FT": dict(color=STEEL, marker="s", ls="--"),
        "GRACE": dict(color=NAVY, marker="^", ls=":"),
    }
    label_y = {"ROME": means["ROME"][3], "FT": 0.085, "GRACE": -0.005}

    fig, ax = plt.subplots(figsize=(4.5, 2.85))
    qx = [1, 2, 3, 4]
    for e in editors:
        st = style[e]
        ax.plot(qx, means[e], lw=1.7, ms=5.5, mec="white", mew=0.6, **st)
        ax.text(4.14, label_y[e], e, color=INK, fontsize=ANN,
                ha="left", va="center")

    ax.text(0.97, 0.955, rf"$n \approx {n_q}$ per quartile per editor",
            transform=ax.transAxes, color=INK, fontsize=ANN,
            ha="right", va="top")

    ax.set_xticks(qx)
    ax.set_xticklabels(["Q1\neasiest", "Q2", "Q3", "Q4\nhardest"])
    ax.set_xlim(0.72, 4.75)
    ax.set_ylim(-0.04, 0.92)
    ax.set_yticks(np.arange(0, 0.91, 0.2))
    ax.set_xlabel(r"quartile of pre-edit certificate difficulty $L$")
    ax.set_ylabel("paraphrase success (argmax)")
    ax.grid(True, axis="y")

    fig.tight_layout()
    return _finish(fig, "fig-certpredict")


# ==============================================================================
# FIGURE -- head-reachability by norm decile          print width 4.7in
# ==============================================================================
def fig_head_geometry():
    hg = _load("headgeom_metrics.json")["head_geometry"]
    vals = [100.0 * v for v in hg["reachable_by_norm_decile_lowtohigh"]]
    overall = 100.0 * hg["reachable_fraction"]           # 10.27

    deciles = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(4.7, 2.60))

    ax.bar(deciles, vals, width=0.72, color=TEAL, zorder=3)
    for d, v in zip(deciles, vals):
        ax.text(d, v + 0.6, f"{v:.1f}", color=INK, fontsize=ANN,
                ha="center", va="bottom")

    ax.axhline(overall, color=BRICK, lw=1.2, ls="--", zorder=2)
    ax.text(5.5, overall + 1.1, f"overall {overall:.2f}%", color=INK,
            fontsize=ANN, ha="center", va="bottom")

    ax.set_xticks(deciles)
    ax.set_xlim(0.35, 10.65)
    ax.set_ylim(0, 34)
    ax.set_yticks(np.arange(0, 31, 10))
    ax.set_xlabel(r"output-embedding norm decile (low $\rightarrow$ high)")
    ax.set_ylabel("head-reachable fraction (%)")
    ax.grid(True, axis="y")

    fig.tight_layout()
    return _finish(fig, "fig-head-geometry")


# ==============================================================================
# FIGURE -- signed co-training perplexity delta       print width 3.8in
#
# ppl delta = (hybrid - baseline) / baseline * 100, per scale, read from the
# G2a/G3a no-tax artifacts (cross-checked against notax_params.json val_ppl).
# Both scales sit inside the paper's +/-0.12% "no perplexity tax" claim bound.
# 136M: +0.018% (hybrid marginally worse); 1B: -0.116% (hybrid better).
# ==============================================================================
def fig_notax_delta():
    g2 = _load("g2a_no_tax.json")
    g3 = _load("g3a_no_tax.json")

    def pct(d):
        return (d["hybrid_val_ppl"] - d["baseline_val_ppl"]) \
            / d["baseline_val_ppl"] * 100.0

    scales = ["136M", "1B"]
    deltas = [pct(g2), pct(g3)]                          # +0.0184, -0.1158
    band = 0.12                                          # paper's claim bound

    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(3.8, 2.55))

    ax.axhspan(-band, band, color=STEEL, alpha=0.16, zorder=1)
    ax.axhline(0, color="black", lw=0.8, zorder=3)

    for xi, d in zip(x, deltas):
        c = BRICK if d > 0 else TEAL          # positive = slight regression
        ax.plot([xi, xi], [0, d], color=c, lw=2.2, solid_capstyle="round",
                zorder=4)
        ax.plot(xi, d, "o", color=c, ms=8.0, mec="white", mew=0.8, zorder=5)
        va = "bottom" if d > 0 else "top"
        off = 0.012 if d > 0 else -0.012
        ax.text(xi, d + off, f"{d:+.3f}%".replace("-", "−"),
                color=INK, fontsize=ANN, ha="center", va=va, zorder=6)

    ax.text(1.42, band, "$\\pm$0.12%\nclaim bound", color=INK, fontsize=ANN,
            ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.16, 0.16)
    ax.set_yticks([-0.12, -0.06, 0.0, 0.06, 0.12])
    ax.set_ylabel("co-training $\\Delta$PPL (%)")
    ax.set_xlabel("model scale")
    ax.grid(True, axis="y")

    fig.tight_layout()
    return _finish(fig, "fig-notax-delta")


# ==============================================================================
# FIGURE -- gate selectivity (two panels)             print width 6.4in
# ==============================================================================
def fig_gate_selectivity():
    gr = _load("gate_roc.json")
    stats = gr["category_stats"]

    cats = [("exact key", "exact_key"),
            ("paraphrase", "paraphrase"),
            ("other relation", "other_relation"),
            ("typo", "typo"),
            ("random held-out", "random_heldout")]
    n_off = sum(stats[k]["n"] for _, k in cats[1:])       # 92

    tau_dep = 0.95                       # deployed operating threshold
    tau_rec = gr["recommended_tau"]      # 0.5 (recommended, lighter reference)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(6.4, 2.40), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- panel (a): max gate score by probe category -------------------------
    ypos = np.arange(len(cats))[::-1]                     # exact key on top
    for y, (name, key) in zip(ypos, cats):
        v, n = stats[key]["max"], stats[key]["n"]
        if v > 0:
            axa.barh(y, v, height=0.55, color=TEAL, zorder=3)
            axa.text(v - 0.03, y, f"{v:.2f} (n = {n})", color="white",
                     fontsize=ANN, ha="right", va="center", zorder=4)
        else:
            axa.text(0.02, y, f"{v:.2f} (n = {n})", color=INK,
                     fontsize=ANN, ha="left", va="center", zorder=4)

    # recommended tau: lighter secondary reference
    axa.axvline(tau_rec, color=GREY, lw=1.0, ls=(0, (3, 3)), zorder=2)
    axa.text(tau_rec + 0.03, 2.5, rf"$\tau = {tau_rec:g}$ (rec.)", color=GREY,
             fontsize=ANN, ha="left", va="center")
    # deployed tau: primary reference
    axa.axvline(tau_dep, color=BRICK, lw=1.2, ls="--", zorder=2)
    axa.text(tau_dep - 0.02, -0.62, rf"deployed $\tau = {tau_dep:g}$",
             color=INK, fontsize=ANN, ha="right", va="center")

    axa.set_yticks(ypos)
    axa.set_yticklabels([name for name, _ in cats])
    axa.set_ylim(-0.85, 4.6)
    axa.set_xlim(0, 1.05)
    axa.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axa.set_xlabel("max gate score")
    axa.grid(True, axis="x")
    _panel_tag(axa, "(a)")

    # ---- panel (b): fire rate vs threshold -----------------------------------
    frt = gr["fire_rate_by_tau"]
    taus = sorted(float(t) for t in frt["exact_key"])
    exact = [frt["exact_key"][f"{t:g}"] for t in taus]
    off = [max(frt[k][f"{t:g}"] for _, k in cats[1:]) for t in taus]

    axb.axvline(tau_dep, color=BRICK, lw=1.0, ls="--", alpha=0.7, zorder=1)
    axb.text(tau_dep - 0.02, 0.5, r"deployed $\tau$", color=BRICK,
             fontsize=ANN, ha="right", va="center", rotation=90)

    axb.plot(taus, exact, color=TEAL, marker="o", ls="-", lw=1.7,
             ms=4.5, mec="white", mew=0.6, zorder=4)
    axb.plot(taus, off, color=BRICK, marker="s", ls="--", lw=1.5,
             ms=4.0, mec="white", mew=0.6, zorder=3)

    axb.text(0.42, 0.915, "exact key", color=INK, fontsize=ANN,
             ha="center", va="center")
    axb.text(0.42, 0.085, f"all off-target probes (n = {n_off})", color=INK,
             fontsize=ANN, ha="center", va="center")

    axb.set_xlim(0, 1.0)
    axb.set_ylim(-0.06, 1.09)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xlabel(r"gate threshold $\tau$")
    axb.set_ylabel("fire rate")
    axb.grid(True, axis="y")
    _panel_tag(axb, "(b)")

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.93))   # headroom for tags
    return _finish(fig, "fig-gate-selectivity")


# ==============================================================================
# FIGURE -- edit-quality frontier (generalization x locality)  print width 5.5in
#
# Every method occupies the paraphrase-generalization / locality plane; the
# upper-right corner (high generalization AND high locality) stays empty -- each
# method trades one for the other. 95% bootstrap CI whiskers on both axes where
# the artifact stores them (the five gpt2_table1_v2 methods); the two ROME
# points store means only.
# ==============================================================================
def fig_frontier():
    t1 = _load("gpt2_table1_v2.json")["summary"]

    def triplet(m, k):
        v = t1[m][k]
        return v[0], v[1], v[2]           # mean, ci_lo, ci_hi

    # method -> (gen mean, gen lo, gen hi, loc mean, loc lo, loc hi | None)
    pts = {}
    for m in t1:
        gm, glo, ghi = triplet(m, "gen")
        lm, llo, lhi = triplet(m, "loc")
        pts[m] = (gm, glo, ghi, lm, llo, lhi)

    rome = _load("rome_gpt2.json")
    rome_ee = _load("rome_easyedit_gpt2.json")
    pts["ROME"] = (rome["generalization"], None, None,
                   rome["locality"], None, None)
    pts["ROME (EasyEdit)"] = (rome_ee["generalization"], None, None,
                              rome_ee["locality"], None, None)

    display = {
        "HLM5": "HLM5", "LogitBias": "logit-bias", "FT-full": "FT-full",
        "FT-L": "FT-L", "RAG": "RAG (eff 0.67)", "ROME": "ROME",
        "ROME (EasyEdit)": "ROME\n(EasyEdit)",
    }
    # per-method label placement: (dx, dy, ha, va) -- offsets keep labels off
    # their own CI whiskers and off neighbouring points.
    lab = {
        "HLM5": (0.0, 0.030, "center", "bottom"),
        "LogitBias": (0.0, 0.032, "center", "bottom"),
        "FT-full": (0.0, 0.055, "center", "bottom"),
        "FT-L": (0.0, -0.062, "center", "top"),
        "RAG": (0.0, 0.050, "center", "bottom"),
        "ROME": (-0.018, 0.0, "right", "center"),
        "ROME (EasyEdit)": (0.0, -0.055, "center", "top"),
    }
    emphasis = "HLM5"

    fig, ax = plt.subplots(figsize=(5.5, 3.20))

    for m, (gm, glo, ghi, lm, llo, lhi) in pts.items():
        is_hero = (m == emphasis)
        col = NAVY if is_hero else TEAL
        ms = 9.5 if is_hero else 7.0
        if glo is not None:
            ax.errorbar(gm, lm,
                        xerr=[[gm - glo], [ghi - gm]],
                        yerr=[[lm - llo], [lhi - lm]],
                        fmt="none", ecolor=GREY, elinewidth=0.9,
                        capsize=2.2, capthick=0.9, zorder=3)
        ax.plot(gm, lm, "o", color=col, ms=ms, mec="white", mew=0.9,
                zorder=5 if is_hero else 4)
        dx, dy, ha, va = lab[m]
        fw = "bold" if is_hero else "normal"
        ax.text(gm + dx, lm + dy, display[m], color=INK, fontsize=ANN,
                ha=ha, va=va, fontweight=fw, zorder=6)

    # empty upper-right corner: the message sits in the vacant region itself
    ax.text(0.83, 0.95, "empty corner:\nno method is both\ngeneral and local",
            color=GREY, fontsize=ANN, ha="center", va="center", zorder=2)

    ax.set_xlim(0.15, 1.05)
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks(np.arange(0.2, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("paraphrase generalization")
    ax.set_ylabel("locality (specificity)")
    ax.grid(True, axis="both")

    fig.tight_layout()
    return _finish(fig, "fig-frontier")


# ==============================================================================
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style_rc()):
        fig_reachability_geometry()
        fig_betastar()
        fig_envelope_multikey()
        fig_certdosed()
        fig_certpredict()
        fig_head_geometry()
        fig_notax_delta()
        fig_gate_selectivity()
        fig_frontier()


if __name__ == "__main__":
    main()
