#!/usr/bin/env python3
"""make_schematics.py -- the two hand-made schematic figures, from source.

The HLM5 paper ships two conceptual schematics that were originally drawn by
hand and had no committed source: ``fig1-pipeline-a-b`` (the two-pipeline
training contract) and ``fig-arch-forward`` (the forward pass with the gated
memory equations). This module regenerates both as deterministic matplotlib
vector PDFs so the community can regenerate or fix them.

Shared camera-ready style with ``scripts/make_figures.py`` (serif + Computer
Modern mathtext, 9pt/8pt print sizes, neutral ink, navy/teal/steel palette):
``style_rc`` and the palette are imported from that module so the schematics
never drift from the data figures. The one extra colour is ``OCHRE`` for the
"same frozen trunk" bridge and the additive-write path.

Each schematic is drawn in inch-space -- the single Axes fills the whole figure
and the data limits equal the figure's physical size in inches, so 1 data unit
== 1 inch on both axes (circles stay round, no aspect surprises). The figures
are saved at their final print width so LaTeX includes them at scale ~1.00:

  figure                design size    included at
  --------------------- -------------- --------------------------------------
  fig1-pipeline-a-b     6.2 x 2.85 in  0.95\\linewidth  (~6.18in, 1-in margins)
  fig-arch-forward      6.4 x 3.05 in  0.98\\linewidth  (~6.37in)

A 200-dpi PNG preview of each basename is written to ``FIGURES_DIR/_preview``.

Run:  python scripts/make_schematics.py   (repo root, CPU-only, no network)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Share the exact house style + palette with the data-figure generator (sibling
# module). Fall back to a local copy of the style block if it cannot be imported
# (e.g. run in isolation without the results/ tree that make_figures touches).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from make_figures import (  # type: ignore
        FIGURES_DIR, PREVIEW_DIR, style_rc,
        NAVY, TEAL, STEEL, BRICK, GREY, INK,
    )
except Exception:  # pragma: no cover - defensive fallback
    from hlm5.io import FIGURES_DIR
    PREVIEW_DIR = FIGURES_DIR / "_preview"
    NAVY, TEAL, STEEL, BRICK, GREY, INK = (
        "#1F3A5F", "#2E6F6A", "#5B7C99", "#A93C2E", "#8A8F94", "#222222")

    def style_rc():
        return {
            "font.family": "serif", "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm", "font.size": 9.0,
            "text.color": INK, "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }

OCHRE = "#9A6419"   # "same frozen trunk" bridge / additive-write path


# ---- small drawing helpers ---------------------------------------------------
def _tint(color, f):
    """Mix ``color`` toward white by fraction ``f`` (0 = color, 1 = white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _new_axes(w, h):
    """A figure of physical size ``w`` x ``h`` inches whose single Axes fills it
    and uses inch-space data coordinates (1 unit == 1 inch on both axes)."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, rows, *, ec, fc, lw=1.2, rounding=0.055, zorder=3):
    """Rounded box with a vertically-stacked label block.

    ``rows`` is a list of (dy, text, size, weight, color, style): ``dy`` is the
    vertical offset in inches from the box centre (positive = up)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        ec=ec, fc=fc, lw=lw, mutation_aspect=1, zorder=zorder))
    cx, cy = x + w / 2.0, y + h / 2.0
    for dy, text, size, weight, color, style in rows:
        ax.text(cx, cy + dy, text, ha="center", va="center", fontsize=size,
                fontweight=weight, style=style, color=color, zorder=zorder + 2)


def _arrow(ax, p0, p1, *, color=INK, lw=1.6, ls="-", conn=None, mut=11,
           zorder=4):
    kw = dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
              mutation_scale=mut, shrinkA=0, shrinkB=0, zorder=zorder)
    if conn is not None:
        kw["connectionstyle"] = conn
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


# ==============================================================================
# FIGURE 1 -- the two-pipeline training contract          print width 6.2in
# ==============================================================================
def fig_pipeline_a_b():
    W, H = 6.2, 2.85
    fig, ax = _new_axes(W, H)

    # palette tints
    navy_fill = _tint(NAVY, 0.90)     # Pipeline A container
    blue_box = _tint(NAVY, 0.82)      # boxes inside Pipeline A
    teal_fill = _tint(TEAL, 0.91)     # Pipeline B container
    teal_box = _tint(TEAL, 0.80)      # boxes inside Pipeline B

    bh = 0.60                          # inner box height (2-line boxes)
    up, dn = +0.115, -0.125            # title / subtitle offsets

    # ---- containers ----------------------------------------------------------
    LxA, RxA = 0.06, 2.68
    LxB, RxB = 3.20, 6.14
    cy0, cy1 = 0.30, 2.56
    ax.add_patch(FancyBboxPatch(
        (LxA, cy0), RxA - LxA, cy1 - cy0,
        boxstyle="round,pad=0,rounding_size=0.06",
        ec=NAVY, fc=navy_fill, lw=1.3, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (LxB, cy0), RxB - LxB, cy1 - cy0,
        boxstyle="round,pad=0,rounding_size=0.06",
        ec=TEAL, fc=teal_fill, lw=1.3, zorder=1))
    ax.text(LxA + 0.14, cy1 - 0.17, "Pipeline A: language pretraining",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=NAVY, zorder=6)
    ax.text(LxB + 0.14, cy1 - 0.17, "Pipeline B: governed fact operations",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color=TEAL, zorder=6)

    # ---- Pipeline A inner boxes ---------------------------------------------
    row_y = 1.62                        # top row of inner boxes (centres)
    fw_x, fw_w = 0.22, 0.76
    ce_x, ce_w = 1.44, 1.10
    _box(ax, fw_x, row_y - bh / 2, fw_w, bh, [
        (up, "FineWeb", 9, "bold", NAVY, "normal"),
        (dn, "tokens", 8, "normal", GREY, "normal")],
        ec=NAVY, fc=blue_box)
    _box(ax, ce_x, row_y - bh / 2, ce_w, bh, [
        (up, "Cross-entropy", 9, "bold", NAVY, "normal"),
        (dn, "baseline / hybrid", 8, "normal", GREY, "normal")],
        ec=NAVY, fc=blue_box)
    ft_x, ft_w, ft_cy = 1.28, 1.24, 0.78
    _box(ax, ft_x, ft_cy - bh / 2, ft_w, bh, [
        (up, "Frozen trunk", 9, "bold", NAVY, "normal"),
        (dn, "released checkpoint", 8, "normal", GREY, "italic")],
        ec=NAVY, fc=blue_box)
    _arrow(ax, (fw_x + fw_w, row_y), (ce_x, row_y), color=INK)
    _arrow(ax, (ce_x + ce_w / 2, row_y - bh / 2), (ft_x + ft_w / 2, ft_cy + bh / 2),
           color=INK, conn="arc3,rad=0")

    # ---- Pipeline B inner boxes ---------------------------------------------
    fa_x, fa_w = 3.42, 0.92
    ms_x, ms_w = 4.76, 1.26
    _box(ax, fa_x, row_y - bh / 2, fa_w, bh, [
        (up, "Fact", 9, "bold", TEAL, "normal"),
        (dn, "source + key", 8, "normal", GREY, "normal")],
        ec=TEAL, fc=teal_box)
    _box(ax, ms_x, row_y - bh / 2, ms_w, bh, [
        (up, "Memory slot", 9, "bold", TEAL, "normal"),
        (dn, "inject / edit / forget", 8, "normal", GREY, "normal")],
        ec=TEAL, fc=teal_box)
    rc_x, rc_w, rc_cy = 3.50, 2.52, 0.78
    _box(ax, rc_x, rc_cy - bh / 2, rc_w, bh, [
        (up, "Receipt or certified refusal", 9, "bold", TEAL, "normal"),
        (dn, "audit trail, support, trace, rollback", 8, "normal", GREY,
         "normal")],
        ec=TEAL, fc=_tint(TEAL, 0.88))
    _arrow(ax, (fa_x + fa_w, row_y), (ms_x, row_y), color=TEAL)
    _arrow(ax, (ms_x + ms_w / 2, row_y - bh / 2), (rc_x + rc_w * 0.60, rc_cy + bh / 2),
           color=TEAL, conn="arc3,rad=0.18")

    # ---- bridge: same frozen trunk feeds Pipeline B --------------------------
    # arrow hugs the right of the gap; label sits in the left of the gap.
    _arrow(ax, (ft_x + ft_w, ft_cy + 0.06), (fa_x + 0.12, row_y - bh / 2),
           color=OCHRE, lw=1.5, ls=(0, (5, 3)), conn="arc3,rad=-0.26", mut=12)
    ax.text(2.80, 1.02, "same\nfrozen\ntrunk",
            ha="center", va="center", fontsize=8, style="italic",
            color=OCHRE, zorder=6, linespacing=1.2)

    _finish(fig, "fig1-pipeline-a-b")


# ==============================================================================
# FIGURE -- HLM5 forward pass with gated-memory equations   print width 6.4in
# ==============================================================================
def fig_arch_forward():
    W, H = 6.4, 3.05
    fig, ax = _new_axes(W, H)

    blue_box = _tint(NAVY, 0.84)
    tan_box = _tint(OCHRE, 0.86)
    steel_box = _tint(STEEL, 0.82)
    mem_fill = _tint(TEAL, 0.90)
    cert_fill = _tint(STEEL, 0.90)

    top_cy = 2.66                       # top-row box centres
    bh = 0.64
    ytop, ybot = top_cy + bh / 2, top_cy - bh / 2

    # ---- top row: Tokens -> trunk -> h -> write -> head -> logits ------------
    tk_x, tk_w = 0.08, 0.84
    tr_x, tr_w = 1.08, 1.32
    h_cx, h_r = 2.66, 0.14
    aw_x, aw_w = 2.94, 1.42
    th_x, th_w = 4.52, 0.88
    lg_x, lg_w = 5.60, 0.70

    _box(ax, tk_x, ybot, tk_w, bh, [
        (+0.11, "Tokens", 9, "bold", INK, "normal"),
        (-0.13, r"$x_{\leq t}$", 9, "normal", GREY, "normal")],
        ec=GREY, fc="white")
    _box(ax, tr_x, ybot, tr_w, bh, [
        (+0.17, "Transformer trunk", 9, "bold", NAVY, "normal"),
        (-0.02, r"$h = f_{\theta}(x)$", 9, "normal", INK, "normal"),
        (-0.19, "frozen, tied head", 8, "normal", STEEL, "italic")],
        ec=NAVY, fc=blue_box)
    ax.add_patch(Circle((h_cx, top_cy), h_r, ec=NAVY, fc="white", lw=1.3,
                        zorder=3))
    ax.text(h_cx, top_cy, r"$h$", ha="center", va="center", fontsize=9,
            color=NAVY, zorder=6)
    _box(ax, aw_x, ybot, aw_w, bh, [
        (+0.17, "Additive write", 9, "bold", OCHRE, "normal"),
        (-0.02, r"$h' = h + \beta\, g(q)\, r$", 9, "normal", INK, "normal"),
        (-0.19, r"off-support: $h' = h$", 8, "normal", OCHRE, "italic")],
        ec=OCHRE, fc=tan_box)
    _box(ax, th_x, ybot, th_w, bh, [
        (+0.11, "Tied head", 9, "bold", STEEL, "normal"),
        (-0.13, r"$z' = W h'$", 9, "normal", INK, "normal")],
        ec=STEEL, fc=steel_box)
    _box(ax, lg_x, ybot, lg_w, bh, [
        (+0.11, "logits", 9, "bold", INK, "normal"),
        (-0.13, r"$\mathrm{arg\,max}$", 8, "normal", GREY, "normal")],
        ec=GREY, fc="white")

    _arrow(ax, (tk_x + tk_w, top_cy), (tr_x, top_cy), color=INK)
    _arrow(ax, (tr_x + tr_w, top_cy), (h_cx - h_r, top_cy), color=INK)
    _arrow(ax, (h_cx + h_r, top_cy), (aw_x, top_cy), color=INK)
    _arrow(ax, (aw_x + aw_w, top_cy), (th_x, top_cy), color=INK)
    _arrow(ax, (th_x + th_w, top_cy), (lg_x, top_cy), color=INK)

    # ---- external gated memory ----------------------------------------------
    mem_x, mem_w = 1.70, 3.05
    mem_top, mem_bot = 2.02, 1.00
    ax.add_patch(FancyBboxPatch(
        (mem_x, mem_bot), mem_w, mem_top - mem_bot,
        boxstyle="round,pad=0,rounding_size=0.05",
        ec=TEAL, fc=mem_fill, lw=1.3, zorder=3))
    mcx = mem_x + mem_w / 2.0
    mem_rows = [
        (1.90, "External gated memory", 9.5, "bold", TEAL, "normal"),
        (1.75, "inject / edit / forget", 8, "normal", TEAL, "italic"),
        (1.58, r"$q = (h - \mu)\, A$", 8.5, "normal", INK, "normal"),
        (1.42, r"$s_i = \mathrm{active}_i\, \mathrm{relu}(\alpha_i)\,"
               r"\max(0,\, \cos(q, k_i))^{\kappa}$", 8.5, "normal", INK,
         "normal"),
        (1.26, r"$r = \mathrm{softmax}(s / T)\, V$", 8.5, "normal", INK,
         "normal"),
        (1.10, r"$g(q) = \max_i s_i \ \ \mathrm{if}\ \max_i s_i \geq \tau,"
               r"\ \ \mathrm{else}\ 0$", 8.5, "normal", INK, "normal"),
    ]
    for yy, text, size, weight, color, style in mem_rows:
        ax.text(mcx, yy, text, ha="center", va="center", fontsize=size,
                fontweight=weight, style=style, color=color, zorder=6)

    # dashed feed-in (unperturbed h) and additive-write return path
    _arrow(ax, (h_cx, top_cy - h_r), (h_cx, mem_top + 0.02),
           color=STEEL, lw=1.3, ls=(0, (4, 3)))
    ax.text(h_cx + 0.12, (top_cy - h_r + mem_top) / 2.0, r"unperturbed $h$",
            ha="left", va="center", fontsize=8, style="italic", color=STEEL,
            zorder=6)
    _arrow(ax, (aw_x + aw_w / 2, mem_top + 0.02), (aw_x + aw_w / 2, ybot),
           color=OCHRE, lw=1.6)
    ax.text(aw_x + aw_w / 2 + 0.10, (mem_top + ybot) / 2.0,
            r"$\beta\, g(q)\, r$", ha="left", va="center", fontsize=8.5,
            color=OCHRE, zorder=6)

    # ---- "why it is certifiable" banner -------------------------------------
    cb_x, cb_w = 0.12, 6.16
    cb_bot, cb_top = 0.12, 0.86
    ax.add_patch(FancyBboxPatch(
        (cb_x, cb_bot), cb_w, cb_top - cb_bot,
        boxstyle="round,pad=0,rounding_size=0.04",
        ec=STEEL, fc=cert_fill, lw=1.1, zorder=2))
    ccx = cb_x + cb_w / 2.0
    ax.text(ccx, 0.74, "Why it is certifiable", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=NAVY, zorder=6)
    ax.text(ccx, 0.53, r"$g$ and $r$ depend only on the unperturbed $h$, so "
            r"every logit margin is affine in the edit strength "
            r"$\beta_{\mathrm{eff}}$:", ha="center", va="center", fontsize=8.5,
            color=INK, zorder=6)
    ax.text(ccx, 0.30, r"$m_{yj}(\beta_{\mathrm{eff}}) = (W_y - W_j)^{\top} h'"
            r" = a_j + \beta_{\mathrm{eff}}\, b_j$   (Lemma 1).", ha="center",
            va="center", fontsize=8.5, color=INK, zorder=6)

    _finish(fig, "fig-arch-forward")


# ==============================================================================
def _finish(fig, stem):
    """Save at exact design size (PDF) plus a 200-dpi PNG preview."""
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = PREVIEW_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style_rc()):
        fig_pipeline_a_b()
        fig_arch_forward()


if __name__ == "__main__":
    main()
