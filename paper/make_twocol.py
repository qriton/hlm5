#!/usr/bin/env python3
"""Generate the two-column variant of the HLM5 paper from the single-column master.

The single-column source ``hlm5-paper.tex`` is the one authored file; the
two-column preprint variant ``hlm5-paper-twocol.tex`` is *derived* from it by a
deterministic transform so the two never drift:

  * the document class becomes ``[10pt,twocolumn]``;
  * ``\\maketitle`` + the ``abstract`` environment are replaced by a
    full-width ``\\twocolumn[...]`` banner that reuses the master's title,
    author and *verbatim* abstract text (so a body-diff shows only the banner
    wrapper and the float changes below);
  * every float is classified as page-spanning (``figure*``/``table*``, art at
    ``\\textwidth``) or column-width (``figure``/``table``, art at
    ``\\columnwidth``) via the fixed map below, keyed by ``\\label``;
  * a handful of display equations that fit the master's ``\\textwidth`` but
    overflow the narrower twocolumn ``\\columnwidth`` are rewritten (line
    break via ``aligned``, or ``\\small``) via ``EQUATION_FIXES`` below, keyed
    by exact equation source so the map stays valid across unrelated edits.

The transform is pure text and deterministic: same input => same output.
Run standalone (``python make_twocol.py``) or via ``build.py``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Float placement map ----------------------------------------------------
# Page-spanning floats: starred environment, artwork at \textwidth.
STAR = {
    # figures that span both columns
    "fig:pipeline", "fig:arch", "fig:envmultikey", "fig:frontier", "fig:certdosed",
    # data / paragraph tables too wide for a single column
    "tab:crosswalk", "tab:beyondargmax", "tab:main", "tab:notax", "tab:energylang",
    "tab:energyops", "tab:ksweep", "tab:gpt2", "tab:certpredict", "tab:trunk",
    "tab:governance", "tab:certpredict18", "app:facts",
}
# Column-width floats: plain environment, artwork at \columnwidth.
COLUMN = {
    "fig:headgeom", "fig:reach", "fig:betastar", "fig:gate", "fig:notax",
    "fig:certpredictfig",
    "tab:envelope",
}

_FLOAT_RE = re.compile(r"\\begin\{(figure|table)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", re.S)
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}")

# --- Display-equation overflow fixes (twocol only) --------------------------
# In the single-column master these displays fit \textwidth comfortably; at
# twocolumn's narrower \columnwidth six of them go overfull (tectonic run
# against hlm5-paper-twocol.tex, 10pt/twocolumn). Each entry maps the exact
# equation source (verbatim substring of the master, unaffected by the float/
# banner transforms above) to a twocol-only replacement. Keyed by equation
# source, not line numbers, so this survives edits elsewhere in the paper.
# Fix chosen per the overfull amount at the time this map was written:
#   <=20pt overfull  -> wrap in \small (mild)
#   >20pt overfull with a natural break point (a relation or \quad/\qquad
#     joining two sub-statements) -> split into \begin{aligned}...\end{aligned}
#     at that break point (preferred over resizebox: keeps the type size and
#     the equation reads top-to-bottom instead of being visually squeezed)
EQUATION_FIXES = [
    # LoRA update rule -- 18.9pt overfull; mild -> \small.
    (
        "\\begin{equation}\n"
        "W_0' = W_0 + \\tfrac{\\alpha}{r}\\,BA,\\quad B\\in\\R^{d\\times r},\\ A\\in\\R^{r\\times k},\\ r\\ll\\min(d,k).\n"
        "\\end{equation}",
        "\\begin{equation}\n"
        "\\small\n"
        "W_0' = W_0 + \\tfrac{\\alpha}{r}\\,BA,\\quad B\\in\\R^{d\\times r},\\ A\\in\\R^{r\\times k},\\ r\\ll\\min(d,k).\n"
        "\\end{equation}",
    ),
    # eq:score, gated kernel score -- 5.4pt overfull; mild -> \small.
    (
        "\\begin{equation}\n"
        "s_i(q)=\\text{active}_i\\cdot\\relu(\\alpha_i)\\cdot\\max\\!\\big(0,\\cos(q,k_i)\\big)^{\\kappa},\\qquad \\kappa=5,\n"
        "\\label{eq:score}\n"
        "\\end{equation}",
        "\\begin{equation}\n"
        "\\small\n"
        "s_i(q)=\\text{active}_i\\cdot\\relu(\\alpha_i)\\cdot\\max\\!\\big(0,\\cos(q,k_i)\\big)^{\\kappa},\\qquad \\kappa=5,\n"
        "\\label{eq:score}\n"
        "\\end{equation}",
    ),
    # eq:hook, gate + write -- 87.3pt overfull; natural break before the
    # \qquad separating the gate definition from the write/readout update.
    (
        "\\begin{equation}\n"
        "g(q)=\\begin{cases}\\max_i s_i(q) & \\max_i s_i(q)\\ge\\tau\\\\ 0 & \\text{otherwise,}\\end{cases}\n"
        "\\qquad h'=h+\\beta\\,g(q)\\,r(q),\\quad z'=Wh',\n"
        "\\label{eq:hook}\n"
        "\\end{equation}",
        "\\begin{equation}\n"
        "\\begin{aligned}\n"
        "g(q)&=\\begin{cases}\\max_i s_i(q) & \\max_i s_i(q)\\ge\\tau\\\\ 0 & \\text{otherwise,}\\end{cases}\\\\\n"
        "h'&=h+\\beta\\,g(q)\\,r(q),\\quad z'=Wh',\n"
        "\\end{aligned}\n"
        "\\label{eq:hook}\n"
        "\\end{equation}",
    ),
    # lem:affine margin decomposition -- 120.2pt overfull (the worst); natural
    # break at the \quad separating the margin equality from the a_j,b_j defs.
    (
        "\\[\n"
        "m_{yj}(\\beta_{\\mathrm{eff}}) = (W_y-W_j)^{\\top}h' = a_j + \\beta_{\\mathrm{eff}}\\,b_j,\n"
        "\\quad a_j=(W_y-W_j)^{\\top}h,\\ \\ b_j=(W_y-W_j)^{\\top}\\bar r .\n"
        "\\]",
        "\\[\n"
        "\\begin{aligned}\n"
        "m_{yj}(\\beta_{\\mathrm{eff}}) &= (W_y-W_j)^{\\top}h' = a_j + \\beta_{\\mathrm{eff}}\\,b_j,\\\\\n"
        "a_j&=(W_y-W_j)^{\\top}h,\\ \\ b_j=(W_y-W_j)^{\\top}\\bar r .\n"
        "\\end{aligned}\n"
        "\\]",
    ),
    # Robust margin lower bound -- 6.2pt overfull; mild -> \small.
    (
        "\\[ m_{yj}(\\beta)\\ \\ge\\ a_j+\\beta b_j-c_j\\,(\\varepsilon_h+\\beta\\,\\varepsilon_r),\n"
        "\\qquad c_j=\\lVert W_y-W_j\\rVert. \\]",
        "{\\small\n"
        "\\[ m_{yj}(\\beta)\\ \\ge\\ a_j+\\beta b_j-c_j\\,(\\varepsilon_h+\\beta\\,\\varepsilon_r),\n"
        "\\qquad c_j=\\lVert W_y-W_j\\rVert. \\]\n"
        "}",
    ),
    # Robust feasible interval L_rob/U_rob -- 44.1pt overfull; natural break at
    # the \qquad already separating the L_rob and U_rob definitions.
    (
        "\\begin{equation*}\n"
        "L_{\\mathrm{rob}}=\\max\\!\\Big(0,\\ \\max_{j:\\,b_j>c_j\\varepsilon_r}\\tfrac{c_j\\varepsilon_h-a_j}{b_j-c_j\\varepsilon_r}\\Big),\n"
        "\\qquad\n"
        "U_{\\mathrm{rob}}=\\min_{j:\\,b_j<c_j\\varepsilon_r}\\tfrac{c_j\\varepsilon_h-a_j}{b_j-c_j\\varepsilon_r},\n"
        "\\end{equation*}",
        "\\begin{equation*}\n"
        "\\begin{aligned}\n"
        "L_{\\mathrm{rob}}&=\\max\\!\\Big(0,\\ \\max_{j:\\,b_j>c_j\\varepsilon_r}\\tfrac{c_j\\varepsilon_h-a_j}{b_j-c_j\\varepsilon_r}\\Big),\\\\\n"
        "U_{\\mathrm{rob}}&=\\min_{j:\\,b_j<c_j\\varepsilon_r}\\tfrac{c_j\\varepsilon_h-a_j}{b_j-c_j\\varepsilon_r},\n"
        "\\end{aligned}\n"
        "\\end{equation*}",
    ),
]


def _fix_overfull_equations(out: str) -> str:
    """Apply the twocol-only display-equation overflow fixes above.

    Each fix's ``old`` text must appear exactly once (a stale/edited master
    equation would make the map silently no-op otherwise, which defeats the
    point of keying by source rather than line number).
    """
    for old, new in EQUATION_FIXES:
        n = out.count(old)
        if n != 1:
            raise SystemExit(
                "make_twocol: EQUATION_FIXES entry not found exactly once "
                f"(found {n}); source snippet starting {old[:60]!r} -- "
                "update the map to match the current master equation."
            )
        out = out.replace(old, new, 1)
    return out


def _strip_textbf(s: str) -> str:
    """Return the inner text of a leading ``\\textbf{...}`` wrapper, if present."""
    s = s.strip()
    m = re.match(r"\\textbf\{(.*)\}\s*$", s, re.S)
    return m.group(1) if m else s


def _build_banner(src: str) -> str:
    """Full-width title/abstract banner reusing the master's exact fields."""
    title = _strip_textbf(re.search(r"\\title\{(.*?)\}\s*\n\s*\\author", src, re.S).group(1))
    author = re.search(r"\\author\[1\]\{([^}]*)\}", src).group(1).strip()
    affil = re.search(r"\\affil\[1\]\{(.*?)\}\s*\n", src, re.S).group(1).strip()
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
    author_line = f"{author} \\quad {affil}"
    # Note: size switches close with \par *inside* the group (correct baseline
    # for the enlarged font) and inter-block spacing uses \vspace, not
    # \\[dim] -- the latter is fragile after a group close inside center.
    return (
        "\\twocolumn[%\n"
        "\\begin{center}\n"
        "{\\LARGE\\bfseries " + title + "\\par}\n"
        "\\vspace{0.6em}\n"
        "{\\large " + author_line + "\\par}\n"
        "\\vspace{1.2em}\n"
        "\\end{center}\n"
        "\\begin{center}\\begin{minipage}{0.93\\textwidth}\n"
        "\\small\\noindent\\textbf{Abstract.}\\quad\n"
        + abstract + "\n"
        "\\end{minipage}\\end{center}\n"
        "\\vspace{1.4em}]\n"
    )


def _transform_float(m: re.Match) -> str:
    env, optarg, inner = m.group(1), m.group(2) or "", m.group(3)
    labels = _LABEL_RE.findall(inner)
    label = next((l for l in labels if l in STAR or l in COLUMN), labels[0] if labels else None)

    if label in STAR:
        star, width = True, "\\textwidth"
    elif label in COLUMN:
        star, width = False, "\\columnwidth"
    else:
        # Unknown float: default tables to spanning, figures to column, and warn
        # so the map can be updated rather than silently guessing.
        star = env == "table"
        width = "\\textwidth" if star else "\\columnwidth"
        sys.stderr.write(
            f"[make_twocol] WARNING: unmapped {env} label={label!r}; "
            f"defaulting to {'starred' if star else 'column'}\n"
        )

    inner = _INCLUDE_RE.sub(lambda g: f"\\includegraphics[width={width}]{{{g.group(1)}}}", inner)
    envout = env + "*" if star else env
    return f"\\begin{{{envout}}}{optarg}{inner}\\end{{{envout}}}"


def transform(src: str) -> str:
    """Return the two-column source derived from single-column ``src``."""
    if "\\documentclass[11pt]{article}" not in src:
        raise SystemExit("make_twocol: unexpected \\documentclass in master; aborting")
    out = src.replace("\\documentclass[11pt]{article}",
                      "\\documentclass[10pt,twocolumn]{article}", 1)

    # Replace \maketitle + abstract with the full-width banner.
    banner = _build_banner(src)
    out, n = re.subn(
        r"\\maketitle\s*\\begin\{abstract\}.*?\\end\{abstract\}",
        lambda _m: banner.rstrip("\n"),
        out, count=1, flags=re.S,
    )
    if n != 1:
        raise SystemExit("make_twocol: could not locate \\maketitle+abstract block")

    # Classify every float.
    out = _FLOAT_RE.sub(_transform_float, out)

    # Fix the display equations that overflow \columnwidth at twocolumn.
    out = _fix_overfull_equations(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=os.path.join(HERE, "hlm5-paper.tex"))
    ap.add_argument("--output", default=os.path.join(HERE, "hlm5-paper-twocol.tex"))
    args = ap.parse_args(argv)

    src = open(args.input, encoding="utf-8").read()
    out = transform(src)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)

    n_star = len(re.findall(r"\\begin\{(?:figure|table)\*\}", out))
    n_col = len(re.findall(r"\\begin\{(?:figure|table)\}", out))
    print(f"[make_twocol] wrote {args.output}")
    print(f"[make_twocol]   spanning floats (starred): {n_star}")
    print(f"[make_twocol]   column floats           : {n_col}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
