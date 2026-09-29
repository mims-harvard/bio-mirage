#!/usr/bin/env python
"""Draws Figure 6 (fig_rq3_traces, "Analysis of reasoning traces and rationales", RQ3).

    python figures/reasoning_traces.py [--out_dir <dir>]

Panels a and b are the reference-to-variant base pair BioReason RL and SFT state in their reasoning
traces against the true pair, with Z_Evo2 from the intact DNA, on the 111 held-out single
substitution queries (drawn by trace_base_pairs.draw_matrix_panel). Panel c is one BioReason-Pro
(RL) evidence conflict, read from protein/bioreason_pro/evidence_conflicts_rl/records.jsonl and
checked string by string
against the record before drawing: the inputs with the protein each came from, the opening of the
reasoning trace, and the emitted GO terms. Panel d is the ribosomal share and DEG precision of the
genes C2S-Scale 27B rationales reference, with the cell sentence in expression order and in random
order (drawn by rationale_panels.panel_c).

Panel c's width is fixed and the canvas height is searched for the smallest value at which no text
overflows or collides. Panels a, b and d are checked value by value against
fig_rq3_trace_base_pair_numbers.csv, fig_rq3_trace_base_pair_sft_numbers.csv and
fig_rq3_rationale_panels_numbers.csv in outputs/figures, so run figures/trace_base_pairs.py and
figures/rationale_panels.py first. Reads INPUT_USE_RESULTS_DIR and the GO ontology (GO_OBO). Writes
fig_rq3_traces (pdf, png, svg), its numbers CSV and a LaTeX caption to outputs/figures, or to
--out_dir.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
from matplotlib.textpath import TextToPath

FIG_DIR = os.path.join(os.environ.get("INPUT_USE_OUTPUT_DIR", "outputs"), "figures")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trace_base_pairs as rq3                # noqa: E402
from figure_style import (init_print_style, ax_in, panel_letter, emit, check_overflow,  # noqa: E402
                          check_text_collisions, _mix, _hex2rgb, _rgb2hex, PROTEIN, DNA, CELL,
                          GREY, CTRL, TICK_FS, LEGEND_FS, EDGE_LW)
import rationale_panels as rq2               # noqa: E402
import faulthandler                           # noqa: E402
faulthandler.cancel_dump_traceback_later()    # rationale_panels arms a repeating timer at import
from input_use.metrics import go_dag          # noqa: E402
from input_use.core.config import GO_OBO      # noqa: E402
from input_use.core import paths as RD        # noqa: E402

PAIRS_REC = os.path.join(RD.PROTEIN_EVIDENCE_CONFLICTS["rl"], "records.jsonl")
PAIR_ID = "Q505F1__Q62227"
COND = "conflict_interpro_b"
LINE_NO = 110

STEM = "fig_rq3_traces"

# ---- panel c: the strings, exactly as drawn.  Every one is asserted against the
# record in verify_panel_c() before the panel is drawn.
QUERIED = "Q62227"
PARTNER = "Q505F1"
NAME_QUERIED = "Nuclear receptor subfamily 0 group B member 2"
NAME_PARTNER = "Nuclear receptor subfamily 2 group C member 1"
DNA_BINDING_GO = "GO:0003677"
EXCERPT = (
    "I begin with the InterPro architecture. The N-terminal half contains a canonical nuclear "
    "receptor DNA-binding module: IPR001628 (Zinc finger, nuclear hormone receptor-type, residues "
    "98–173) and IPR013088 (Zinc finger, NHR/GATA-type, residues 93–199) define a tandem "
    "C4 zinc-finger array that recognizes hormone response elements and mediates dimerization and "
    "high-affinity DNA binding."
)
# The claim inside the excerpt that only the partner's InterPro block supports.  Drawn in the
# partner's colour.
PHRASE = "high-affinity DNA binding"
EMITTED = [
    ("Molecular Function", [("GO:0005515", "protein binding")]),
    ("Biological Process", [("GO:0032922", "circadian regulation of gene expression"),
                            ("GO:0007219", "Notch signaling pathway"),
                            ("GO:0045892", "negative regulation of DNA-templated transcription")]),
    ("Cellular Component", [("GO:0032991", "protein-containing complex")]),
]
# The three evidence channels, grouped BY PROTEIN.  verify_panel_c() still checks each channel's
# provenance against records.jsonl line 110 one by one -- only the rendering is grouped.
CHANNELS = [("ESM3 representation", QUERIED),
            ("GO-GPT terms", QUERIED),
            ("InterPro domains", PARTNER)]
CHANNEL_GROUPS = [(QUERIED, "ESM3 representations, GO-GPT"),
                  (PARTNER, "InterPro domains")]
NAME_SUFFIX = {QUERIED: "(no DNA-binding domain)", PARTNER: "(has a DNA-binding domain)"}
FULL_NAME = {QUERIED: NAME_QUERIED, PARTNER: NAME_PARTNER}

# Our annotation at the foot of the emitted GO block, not model output, drawn in italic in the
# queried protein's blue (PROTEIN.model_dark). The emitted set contains GO:0045892 "negative
# regulation of DNA-templated transcription", which names DNA; the sentence stays true because it
# is scoped to GO:0003677 and its descendants, and GO:0045892 is a biological process term outside
# that closure.
GO_NOTE = "No GO:0003677 DNA binding or its descendant"
GO_NOTE_FG = PROTEIN.model_dark
GO_CLOSURE_N = None

SEC_BG = "#F5F6F7"
TRACE_FG = "#1A1A1A"
CAT_FG = "#3D3D3D"


# ================================================================ colour arithmetic
def _hue_shift(base, deg):
    """The same base colour, hue rotated `deg` degrees, at its own lightness and saturation."""
    h, l, s = colorsys.rgb_to_hls(*_hex2rgb(base))
    return _rgb2hex(colorsys.hls_to_rgb((h + deg / 360.0) % 1.0, l, s))


def _luminance(h):
    c = [(x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4) for x in _hex2rgb(h)]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def wcag(a, b):
    """WCAG 2.1 contrast ratio between two hexes."""
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _lab(h):
    c = [(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) for x in _hex2rgb(h)]
    R, G, B = c
    X, Y, Z = (0.4124 * R + 0.3576 * G + 0.1805 * B,
               0.2126 * R + 0.7152 * G + 0.0722 * B,
               0.0193 * R + 0.1192 * G + 0.9505 * B)
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(X / 0.95047), f(Y / 1.0), f(Z / 1.08883)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e76(a, b):
    return sum((x - y) ** 2 for x, y in zip(_lab(a), _lab(b))) ** 0.5


MIN_CONTRAST = 4.5     # WCAG AA for the 5 pt accessions / phrase on SEC_BG
MIN_DELTA_E = 60.0     # separation the two proteins must keep at 5 pt

# Protein B, the queried protein, is the figure's subject and keeps the protein hue.
COL_B = PROTEIN.model_dark                                       # #0B7299, 5.00:1 on SEC_BG
# Protein A, the partner, has to clear BOTH thresholds against SEC_BG and against COL_B.  Nothing
# in the house palette does: PROTEIN.perturbed -- the role that fits the partner semantically,
# since its InterPro block IS the swapped channel -- is 2.23:1 and dE76 31.3, and every other
# PROTEIN and DNA role is either too light or too close; the only palette entry that clears both is
# CELL.model_dark, which would read as the single-cell modality.  So the partner leaves the protein
# hue: the base hue rotated 170 degrees (to red) and taken 20 % to black.  This departs from the
# style module's one hue per modality and is stated in the caption comment; red marks the channel
# that does not belong to the queried protein.
COL_A = _mix(_hue_shift(PROTEIN.model, 170), "#000000", 0.20)    # #AA1E0C
COL = {PARTNER: COL_A, QUERIED: COL_B}
assert wcag(COL_A, SEC_BG) >= MIN_CONTRAST, (COL_A, wcag(COL_A, SEC_BG))
assert wcag(COL_B, SEC_BG) >= MIN_CONTRAST, (COL_B, wcag(COL_B, SEC_BG))
assert delta_e76(COL_A, COL_B) >= MIN_DELTA_E, (COL_A, COL_B, delta_e76(COL_A, COL_B))

ROWS: list[dict] = []
PANEL_C_TEXTS: list = []      # (artist, (left_in, right_in)) for check_panel_c_fits
CSV_KEYS = ["figure", "panel", "arm", "checkpoint", "condition", "group", "x", "series", "value",
            "n", "k", "ci_low", "ci_high", "drawn", "source", "key"]


def rec(**kw):
    ROWS.append({"figure": "rq3_traces", **kw})


# ================================================================ panel c: read and verify
def load_pair_records():
    out = {}
    for i, line in enumerate(open(PAIRS_REC), 1):
        if PAIR_ID not in line:
            continue
        r = json.loads(line)
        if r.get("example_id") == PAIR_ID and r.get("model") == "bioreason_pro_rl":
            out[r["condition"]] = (i, r)
    return out


EM_RE = re.compile(r"^\s*-\s+(Molecular Function|Biological Process|Cellular Component):\s*$")
GO_RE = re.compile(r"^\s*-\s+(GO:\d{7})\s+(.+?)\s*$")


def parse_emitted(raw):
    """The GO block the model emitted, read back out of the generation, not typed in."""
    out, cur = [], None
    for ln in raw.split("\n"):
        m = EM_RE.match(ln)
        if m:
            cur = (m.group(1), [])
            out.append(cur)
            continue
        g = GO_RE.match(ln)
        if g and cur is not None:
            cur[1].append((g.group(1), g.group(2)))
    return out


def verify_panel_c():
    """Assert every string panel c draws against the stored record.  Nothing is drawn until this
    returns, and each check also goes to the CSV so the panel carries its provenance."""
    by_cond = load_pair_records()
    line_no, r = by_cond[COND]
    assert line_no == LINE_NO, (line_no, LINE_NO)
    assert r["record_id"].endswith(f"::bioreason_pro_rl::{PAIR_ID}::{COND}"), r["record_id"]
    a_line, ra = by_cond["aligned_a"]      # every channel from Q505F1
    b_line, rb = by_cond["aligned_b"]      # every channel from Q62227

    iv, gt = r["intervention"], r["ground_truth"]
    checks = [
        ("intervention.embedding_id is the queried protein", iv["embedding_id"] == QUERIED,
         iv["embedding_id"], "intervention.embedding_id"),
        ("intervention.gogpt_id is the queried protein", iv["gogpt_id"] == QUERIED,
         iv["gogpt_id"], "intervention.gogpt_id"),
        ("intervention.interpro_id is the partner (the swapped channel)",
         iv["interpro_id"] == PARTNER, iv["interpro_id"], "intervention.interpro_id"),
        ("the record is flagged a conflict", iv["is_conflict"] is True, iv["is_conflict"],
         "intervention.is_conflict"),
        ("input.sequence is byte-identical to the queried protein's own arm (aligned_b)",
         r["input"]["sequence"] == rb["input"]["sequence"], f"len {len(r['input']['sequence'])}",
         f"input.sequence == line {b_line} input.sequence"),
        ("input.gogpt is byte-identical to the queried protein's own arm (aligned_b)",
         r["input"]["gogpt"] == rb["input"]["gogpt"], "identical",
         f"input.gogpt == line {b_line} input.gogpt"),
        ("input.interpro is byte-identical to the PARTNER's own arm (aligned_a)",
         r["input"]["interpro"] == ra["input"]["interpro"], "identical",
         f"input.interpro == line {a_line} input.interpro"),
        ("input.sequence differs from the partner's sequence",
         r["input"]["sequence"] != ra["input"]["sequence"],
         f"{len(rb['input']['sequence'])} vs {len(ra['input']['sequence'])} aa",
         "input.sequence != aligned_a input.sequence"),
        ("queried protein name", gt["protein_name_b"].startswith(NAME_QUERIED),
         gt["protein_name_b"], "ground_truth.protein_name_b"),
        ("partner protein name", gt["protein_name_a"].startswith(NAME_PARTNER),
         gt["protein_name_a"], "ground_truth.protein_name_a"),
        ("the queried protein lacks a DNA-binding domain",
         gt["spec_b"]["forbid"] == [DNA_BINDING_GO] and gt["spec_b"]["require"] == [],
         gt["spec_b"]["label"], "ground_truth.spec_b"),
        ("the partner has one", gt["spec_a"]["require"] == [DNA_BINDING_GO],
         gt["spec_a"]["label"], "ground_truth.spec_a"),
        ("the trace excerpt appears verbatim in the stored generation",
         EXCERPT in r["output"]["raw"], f"{len(EXCERPT)} characters", "output.raw"),
        ("the excerpt is the opening of the trace",
         r["output"]["parsed"]["think"].strip().startswith(EXCERPT), "startswith",
         "output.parsed.think"),
    ]
    emitted = parse_emitted(r["output"]["raw"])
    checks.append(("the emitted GO block parsed out of the generation matches the drawn one",
                   emitted == EMITTED, "; ".join(f"{c}: " + ", ".join(g for g, _ in t)
                                                 for c, t in emitted), "output.raw GO block"))
    ids = sorted(g for _, terms in EMITTED for g, _ in terms)
    checks.append(("the drawn ids are exactly output.parsed.go_terms",
                   sorted(r["output"]["parsed"]["go_terms"]) == ids, ", ".join(ids),
                   "output.parsed.go_terms"))
    checks.append((f"{DNA_BINDING_GO} (DNA binding) is not among the emitted terms",
                   DNA_BINDING_GO not in r["output"]["parsed"]["go_terms"], "absent",
                   "output.parsed.go_terms"))
    checks.append((f"{DNA_BINDING_GO} is not in the GO-GPT block the model was given",
                   DNA_BINDING_GO not in r["input"]["gogpt"], "absent", "input.gogpt"))
    checks.append(("every emitted id is in the GO-GPT block the model was given",
                   all(g in r["input"]["gogpt"] for g in ids), f"{len(ids)}/{len(ids)}",
                   "input.gogpt"))

    bad = [c for c in checks if not c[1]]
    rows = []
    for name, ok, shown, key in checks:
        print(f"  [panel c] {'ok ' if ok else 'FAIL'}  {name}: {shown}")
        rows.append(dict(panel="c", arm="BioReason-Pro RL (released)", condition=COND,
                         group=PAIR_ID, series=f"verified against the record: {name}",
                         value=str(shown), n=1, drawn=False,
                         source=f"{PAIRS_REC}:{LINE_NO}", key=key))
    assert not bad, [c[0] for c in bad]
    for row in rows:
        rec(**row)
    return r, rows


def verify_go_absent(emitted_ids):
    """Assert the drawn annotation before it is drawn: GO:0003677 DNA binding and its is_a/part_of
    descendant closure must not intersect the terms the model emitted.

    The closure uses go-basic.obo and the is_a + part_of traversal of input_use.metrics.go_dag, the
    same as the BioReason-Pro evidence conflict readouts."""
    global GO_CLOSURE_N
    obo = str(GO_OBO)
    dag = go_dag.GODag(obo)
    closure = dag.descendants(DNA_BINDING_GO, include_self=True)
    mf = {g for g in closure if dag.aspect_of.get(g) == "MF"}
    hit = sorted(set(emitted_ids) & closure)
    GO_CLOSURE_N = len(mf)
    print(f"  [panel c] GO closure of {DNA_BINDING_GO} ({dag.name.get(DNA_BINDING_GO)}): "
          f"{len(closure)} terms by is_a+part_of from {obo}, {len(mf)} of them molecular "
          f"function; intersection with the {len(emitted_ids)} emitted terms: {hit}")
    for g in emitted_ids:
        print(f"      emitted {g} {dag.name.get(g, '?')!r} aspect {dag.aspect_of.get(g, '?')}"
              f"{'  <- names DNA, biological process, NOT in the closure' if g == 'GO:0045892' else ''}")
    assert not hit, hit
    return dict(closure=len(closure), mf=len(mf), note=GO_NOTE)


# ================================================================ text measuring and wrapping
_T2P = TextToPath()


def measure(fig, renderer, s, fs, weight="normal", style="normal"):
    """FONT ADVANCE width of `s` in inches -- the width a backend will actually advance the pen by.

    NOT Text.get_window_extent: that measures the string hinted at the figure's dpi, and this
    figure is measured on a 100 dpi scratch canvas but rendered at 600 dpi (PNG) and as vector
    outlines (PDF/SVG), where the hinted advances differ.  Measured on the trace's last line:
    get_window_extent gives 2.52875 in for the 2.5 in prefix at 100 dpi, 2.51208 at 600 dpi, and
    the unhinted advance is 2.51569 -- so offsets taken at 100 dpi land 0.0131 in (7.8 px at
    600 dpi, over half a word-space at 5 pt) too far right, which is what put a visible gap on each
    side of the highlighted phrase.  `fig` and `renderer` are kept in the signature because every
    call site has them and because the measurement is deliberately independent of both."""
    prop = FontProperties(size=fs, weight=weight, style=style)
    w, _, _ = _T2P.get_text_width_height_descent(s, prop, ismath=False)
    return w / 72.0


def wrap(fig, renderer, s, width_in, fs, weight="normal", style="normal"):
    lines, cur = [], ""
    for word in s.split():
        trial = word if not cur else cur + " " + word
        if measure(fig, renderer, trial, fs, weight, style) <= width_in or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


LS = 1.20          # line spacing multiple used for every text block in panel c


def block_h(n_lines, fs):
    """Height in inches of an n-line text artist drawn with linespacing=LS at fs points."""
    return ((n_lines - 1) * LS * fs + fs) / 72.0


# ================================================================ panel c layout
BODY_FS, HEAD_FS = 5.0, 5.6
SEC_PAD = 0.016          # white space above/below the text inside a section rectangle
SEC_PADX = 0.052         # white space left of the text and right of the last glyph
SEC_GAP = 0.024          # white space between two section rectangles
GRP_GAP = 0.070          # between the two protein groups when they share one line
GAP_HEAD, GAP_CH_PROT, GAP_PROT, GAP_CAT = 0.012, 0.024, 0.012, 0.008
CAT_GAP = 0.055          # between a GO category name and its first term on the same line
TERM_GAP = 0.070         # between two GO terms on the same line
GO_IND = 0.035           # hanging indent of a GO category's continuation line
GAP_NOTE = 0.014         # above OUR annotation line under the emitted block
# Wrapping reserves this much of the text box.  measure() returns the FONT ADVANCE, while
# check_panel_c_fits reads the drawn artist back with get_window_extent, which is the string hinted
# at the figure's 100 dpi and runs up to 0.013 in wide on a 2.8 in line; without the reserve a line
# wrapped exactly to the box is reported as overflowing it.
WRAP_SLACK = 0.020


def trace_segments(fig, renderer, lines, fs, verbose=True):
    """Split each wrapped trace line into coloured segments.

    The excerpt is drawn line by line and, within a line, segment by segment, because PHRASE has to
    carry the partner's colour while the rest of the excerpt stays TRACE_FG.  Each segment's x is
    the ADVANCE width of the line prefix before it (see measure()), so the segments butt exactly
    where one text artist would have put them, and a highlight that straddles a wrap is simply two
    segments on two lines.

    Returns [[(dx_in, text, colour), ...], ...], one list per line."""
    joined = " ".join(lines)
    assert joined == " ".join(EXCERPT.split()), "wrapping changed the excerpt"
    assert joined.count(PHRASE) == 1, joined.count(PHRASE)
    lo = joined.index(PHRASE)
    hi = lo + len(PHRASE)
    out, off = [], 0
    for ln in lines:
        cuts = sorted({0, len(ln)} | {c for c in (lo - off, hi - off) if 0 < c < len(ln)})
        segs = []
        for a, b in zip(cuts, cuts[1:]):
            col = COL_A if (off + a >= lo and off + b <= hi) else TRACE_FG
            dx = measure(fig, renderer, ln[:a], fs) if a else 0.0
            segs.append((dx, ln[a:b], col))
        out.append(segs)
        off += len(ln) + 1
    # VERIFY: the drawn segments, concatenated in draw order with one space per line break, are the
    # wrapped text exactly -- no character gained, lost or reordered by the colour split.
    back = " ".join("".join(s for _, s, _ in segs) for segs in out)
    assert back == joined, (len(back), len(joined))
    if verbose:
        n_hi = sum(1 for segs in out for _, s, c in segs if c == COL_A)
        print(f"  [panel c] trace drawn as {sum(len(s) for s in out)} segments on {len(out)} lines;"
              f" re-concatenation matches the wrapped text ({len(back)} chars); "
              f"{PHRASE!r} in {n_hi} segment(s)")
    return out


def go_flow(fig, renderer, text_w):
    """The 'Emitted GO terms' section laid out INLINE: each category name, then its terms running on
    after it and wrapping, instead of one term per line.  Returns [(y_off_in, segments)], height."""
    pitch = LS * BODY_FS / 72.0
    lines, y, first = [], 0.0, True
    for cat, terms in EMITTED:
        if not first:
            y += pitch + GAP_CAT
        first = False
        segs = [(0.0, cat, dict(fontstyle="italic", color=CAT_FG))]
        x = measure(fig, renderer, cat, BODY_FS, style="italic") + CAT_GAP
        for g, n in terms:
            s = f"{g} {n}"
            w = measure(fig, renderer, s, BODY_FS)
            if segs and x + w > text_w + 1e-9:
                lines.append((y, segs))
                segs, y, x = [], y + pitch, GO_IND
            if x + w > text_w + 1e-9:
                # the term alone is wider than one line: wrap it by words, hanging at GO_IND
                for k, part in enumerate(wrap(fig, renderer, s, text_w - GO_IND, BODY_FS)):
                    if k:
                        lines.append((y, segs))
                        segs, y = [], y + pitch
                    segs.append((GO_IND, part, {}))
                x = GO_IND + measure(fig, renderer, segs[-1][1], BODY_FS) + TERM_GAP
                continue
            segs.append((x, s, {}))
            x += w + TERM_GAP
        lines.append((y, segs))
    return lines, y + block_h(1, BODY_FS)


def panel_c_lines(fig, renderer, box_w, verbose=True):
    """Every wrapped line of panel c, per section, at the panel's text width."""
    text_w = box_w - WRAP_SLACK
    # ---- channels, grouped by protein.  Each group is an accession drawn bold in its own colour
    # followed by its channels; the two groups share ONE line when the panel is wide enough.
    acc_w = {k: measure(fig, renderer, k + ":", BODY_FS, "bold") + 0.030 for k, _ in CHANNEL_GROUPS}
    grp_w = [acc_w[k] + measure(fig, renderer, lab, BODY_FS) for k, lab in CHANNEL_GROUPS]
    one_line = grp_w[0] + GRP_GAP + grp_w[1] <= text_w + 1e-9
    ch_rows = []
    if one_line:
        ch_rows.append([(0.0, CHANNEL_GROUPS[0]), (grp_w[0] + GRP_GAP, CHANNEL_GROUPS[1])])
    else:
        ch_rows = [[(0.0, g)] for g in CHANNEL_GROUPS]
    name_x = measure(fig, renderer, QUERIED, BODY_FS, "bold") + 0.045
    s1 = dict(
        head="Inputs", ch_rows=ch_rows, acc_w=acc_w, ch_one_line=one_line,
        ch_width=grp_w[0] + GRP_GAP + grp_w[1],
        proteins=[(k, wrap(fig, renderer, f"{FULL_NAME[k]} {NAME_SUFFIX[k]}", text_w - name_x,
                           BODY_FS)) for k, _ in CHANNEL_GROUPS],
        name_x=name_x)
    body = wrap(fig, renderer, EXCERPT, text_w, BODY_FS)
    s2 = dict(head="Reasoning trace", body=body,
              segments=trace_segments(fig, renderer, body, BODY_FS, verbose=verbose))
    go_lines, h3 = go_flow(fig, renderer, text_w)
    pitch = LS * BODY_FS / 72.0
    note = wrap(fig, renderer, GO_NOTE, text_w, BODY_FS, style="italic")
    note_y = h3 - block_h(1, BODY_FS) + pitch + GAP_NOTE
    s3 = dict(head="Output GO Terms", go_lines=go_lines, note=note, note_y=note_y)
    h3 = note_y + block_h(len(note), BODY_FS)

    h1 = block_h(len(ch_rows), BODY_FS) + GAP_CH_PROT + \
        sum(block_h(len(w), BODY_FS) + GAP_PROT for _, w in s1["proteins"])
    h2 = block_h(len(body), BODY_FS)
    heads = block_h(1, HEAD_FS) + GAP_HEAD
    secs = [(s1, heads + h1), (s2, heads + h2), (s3, heads + h3)]
    total = sum(h + 2 * SEC_PAD for _, h in secs) + 2 * SEC_GAP
    n_lines = (len(ch_rows) + sum(len(w) for _, w in s1["proteins"]), len(body),
               len(go_lines) + len(note))
    if verbose:
        print(f"  [panel c] channels on {len(ch_rows)} line(s), both groups measuring "
              f"{s1['ch_width']:.4f} in against a {text_w:.4f} in text width")
        print(f"  [panel c] lines: inputs {n_lines[0]}, trace {n_lines[1]}, "
              f"GO {n_lines[2]} (inline); sections "
              f"{[round(h + 2 * SEC_PAD, 3) for _, h in secs]} in, total {total:.3f} in")
    return secs, total, n_lines


def txt(fig, x_in, y_in, s, fs, **kw):
    """Text placed by INCHES from the left and top edges, top-left anchored by default."""
    W_, H_ = fig.get_size_inches()
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    return fig.text(x_in / W_, 1 - y_in / H_, s, fontsize=fs, **kw)


def rect(fig, x_in, y_in, w_in, h_in, **kw):
    W_, H_ = fig.get_size_inches()
    p = Rectangle((x_in / W_, 1 - (y_in + h_in) / H_), w_in / W_, h_in / H_,
                  transform=fig.transFigure, clip_on=False, **kw)
    fig.add_artist(p)
    return p


def draw_panel_c(fig, x0, y0, C_W, secs):
    # The text starts one SEC_PADX in from the rectangle.
    text_x = x0 + SEC_PADX
    text_w = C_W - 2 * SEC_PADX
    PANEL_C_TEXTS.clear()

    def keep(t):
        PANEL_C_TEXTS.append((t, (text_x, text_x + text_w)))
        return t

    y = y0
    for (sec, body_h) in secs:
        h = body_h + 2 * SEC_PAD
        rect(fig, x0, y, C_W, h, facecolor=SEC_BG, edgecolor="none", zorder=0)
        yy = y + SEC_PAD
        keep(txt(fig, text_x, yy, sec["head"], HEAD_FS, fontweight="bold", zorder=3))
        yy += block_h(1, HEAD_FS) + GAP_HEAD
        if "ch_rows" in sec:
            for i, row in enumerate(sec["ch_rows"]):
                ry = yy + i * LS * BODY_FS / 72.0
                for dx, (acc, lab) in row:
                    keep(txt(fig, text_x + dx, ry, acc + ":", BODY_FS, color=COL[acc],
                             fontweight="bold", zorder=3))
                    keep(txt(fig, text_x + dx + sec["acc_w"][acc], ry, lab, BODY_FS, zorder=3))
            yy += block_h(len(sec["ch_rows"]), BODY_FS) + GAP_CH_PROT
            for acc, lines in sec["proteins"]:
                keep(txt(fig, text_x, yy, acc, BODY_FS, color=COL[acc], fontweight="bold",
                         zorder=3))
                keep(txt(fig, text_x + sec["name_x"], yy, "\n".join(lines), BODY_FS,
                         linespacing=LS, zorder=3))
                yy += block_h(len(lines), BODY_FS) + GAP_PROT
        elif "segments" in sec:
            for i, segs in enumerate(sec["segments"]):
                ry = yy + i * LS * BODY_FS / 72.0
                for dx, s, col in segs:
                    keep(txt(fig, text_x + dx, ry, s, BODY_FS, zorder=3, color=col))
        else:
            for y_off, segs in sec["go_lines"]:
                for dx, s, kw in segs:
                    keep(txt(fig, text_x + dx, yy + y_off, s, BODY_FS, zorder=3, **kw))
            keep(txt(fig, text_x, yy + sec["note_y"], "\n".join(sec["note"]), BODY_FS,
                     fontstyle="italic", color=GO_NOTE_FG, linespacing=LS, zorder=3))
        y += h + SEC_GAP
    return y - SEC_GAP


# ================================================================ panels a and b: CSV rows
def record_matrix(letter, ck, M, cfg):
    cols = rq3.PAIRS + [rq3.NONE]
    tot = M.sum(axis=1, keepdims=True)
    F = np.divide(M, tot, out=np.zeros(M.shape, dtype=float), where=tot > 0)
    n_tot = int(M.sum())
    diag = int(sum(M[i, i] for i in range(len(rq3.PAIRS))))
    ch = rq3.chance_diagonal(M)
    for i in range(len(rq3.PAIRS)):
        for j, cj in enumerate(cols):
            if M[i, j]:
                rec(panel=letter, arm=cfg["name"], checkpoint=ck, condition="wt",
                    series=f"drawn cell: true {rq3.PAIRS[i]} -> stated {cj}, fraction of that "
                           f"row's {int(tot[i, 0])} held-out queries",
                    value=round(float(F[i, j]), 4), n=int(tot[i, 0]), k=int(M[i, j]), drawn=True,
                    source=cfg["file"], key="count / row total")
        rec(panel=letter, arm=cfg["name"], checkpoint=ck, condition="wt",
            series=f"row total: held-out single-substitution queries whose true pair is "
                   f"{rq3.PAIRS[i]}",
            value=int(tot[i, 0]), n=n_tot, k=int(tot[i, 0]), drawn=True, source=rq3.EXAMPLES,
            key="row denominator")
    rec(panel=letter, arm=cfg["name"], checkpoint=ck, condition="wt",
        series="title: diagonal total / queries (stated pair equals the true pair)",
        value=round(diag / n_tot, 4), n=n_tot, k=diag, drawn=True, source=cfg["file"],
        key="trace(M[:12,:12])")
    rec(panel=letter, arm=cfg["name"], checkpoint=ck, condition="wt",
        series="expected diagonal if the stated pair were independent of the true pair",
        value=round(ch / n_tot, 4), n=n_tot, k=round(ch, 2), drawn=False, source=cfg["file"],
        key="chance_diagonal()")
    rec(panel=letter, arm=cfg["name"], checkpoint=ck, condition="wt",
        series="queries stating no pair ('none' column total)",
        value=round(int(M[:, -1].sum()) / n_tot, 4), n=n_tot, k=int(M[:, -1].sum()), drawn=True,
        source=cfg["file"], key="M[:, 'none'].sum()")
    top_j = int(np.argmax(M[:, :len(rq3.PAIRS)].sum(axis=0)))
    rec(panel=letter, arm=cfg["name"], checkpoint=ck, condition="wt",
        series="most frequent stated pair, share of all queries",
        value=round(int(M[:, :len(rq3.PAIRS)].sum(axis=0)[top_j]) / n_tot, 4), n=n_tot,
        k=int(M[:, :len(rq3.PAIRS)].sum(axis=0)[top_j]), drawn=False, source=cfg["file"],
        key=rq3.PAIRS[top_j])
    return n_tot, diag


# ================================================================ geometry
W = 7.00                             # set by allocate() from panel c's width
H = 2.00                             # searched: see MIN_H_SEARCH / set_h()
BOT_PAD = 0.02                       # white space kept under the lowest ink
COL_TOP = 0.020                      # column titles FIRST, top-anchored so all share one top
COL_FS = 6.0                         # bold
COL_GAP = 0.043                      # under the column titles, above the panel letters
#   block_h(1, 6.0) is 0.083 in but a 6 pt bold line renders 0.090 in tall, so this
#   leaves ~0.036 in of white between the title glyphs and the letter glyphs.
LET_H = 0.100                        # measured height of an 8 pt bold panel letter
LET_GAP = 0.018                      # under the panel letters
COL_TITLES = True                    # set False to measure what the titles cost in height
TIT = 0.22                           # TWO-line title band over a and b
L_GUT = 0.545                        # a's y label (0.130 in deep, mathtext) + y ticks
GAP_AB, GAP_BC = 0.10, 0.11
GAP_CD_BASE = 0.11                   # widened below if d's column title needs the room
GAP_CD = GAP_CD_BASE
# d's y label gutter: 0.194 in for the two rotated lines + D_LABELPAD + 0.158 in for '100'
# + 0.021 in tick pad + 0.031 in tick length.
D_LABELPAD = 1.5
D_LABEL = 0.42                       # d's y label is TWO rotated lines, plus its y tick labels
D_W_MIN = 0.86                       # d's flat value labels are 0.106 in wide and 0.145*D_W apart
RIGHT = 0.025
CB_H, CB_TOP_GAP = 0.038, 0.035      # shared horizontal colour bar under a and b
CB_W = 1.00
MIN_PITCH = 0.070                    # a/b cell pitch floor: the 5 pt y tick labels are 0.065 in
AX_W_MIN = rq3.NCOL_U * MIN_PITCH
M_H_MAX = 0.97                       # matrix height ceiling; above it a/b, not c, would set H

# Panel c's width.  2.921 in is the result of a descending-grid search at H = 2.00 (the widest
# width whose next step down, 2.916 in, pushes the trace from 5 wrapped lines to 6 and overruns the
# available height by 0.058 in).  Panel c is drawn at 0.80 x 1.20 of that width, and the canvas
# height is searched for it -- see search_h().
C_W_AT_H2 = 2.921
C_SCALE = 0.80
C_W_NARROW = round(C_W_AT_H2 * C_SCALE, 4)          # 2.3368 in, on the 7.00 x 2.610 in canvas
C_SCALE2 = 1.20
C_W_TARGET = round(C_W_NARROW * C_SCALE2, 4)         # 2.8042 in

# a, b and d keep exactly the widths the 7.00 in canvas gave them; the canvas WIDTH absorbs the
# change in panel c, so W is derived rather than fixed.
AX_W_FIX = rq3.NCOL_U * M_H_MAX / len(rq3.PAIRS)     # 1.079125 in
# Panel d's width.  Two binding limits, both measured: the bar width (0.26 x-units of a 2.0-unit
# axis, so 0.13 * D_W; the print-style floor is 0.03 in) and the six flat value labels, which are
# 0.119 in wide as rendered ('79') on centres 0.145 * D_W apart.  Measured at 0.95 in: bars
# 0.1235 in (4.1x the floor), and 0.0190 in of white between the nearest pair of labels.  The
# labels touch at D_W = 0.819 in and the bars hit their floor at 0.231 in, so the labels are the
# binding limit and 0.95 in leaves 0.131 in of further narrowing before they meet.
D_W_FIX = 0.9500                                     # in


def set_h(h):
    global H
    H = h


COL_CLEAR = 0.050        # least white between panel c's block and d's column title
# Least white between panel c's block and d's rotated y axis label.  This is what stops D_SHIFT --
# which slides panel d left under its much wider title -- from closing the c/d gap: it is the
# measured clearance the guard below enforces, so raising it IS "more space between c and d".
D_LAB_CLEAR_MIN = 0.280
D_SHIFT = 0.0            # panel d's axes slide LEFT by this, measured by measure_d_shift()


def d_title_span():
    """Width available to d's column title: from COL_CLEAR right of panel c's block to the right
    margin.  It reduces to GAP_CD + D_LABEL + D_W - COL_CLEAR, so GAP_CD is the only lever."""
    return GAP_CD + D_LABEL + D_W_FIX - COL_CLEAR


def widen_gap_cd_for_d_title():
    """Set GAP_CD so d's column title fits on ONE line.  It is 1.870 in at 6 pt bold against panel
    d's 1.195 in, so it runs left over the c/d gap; when even the whole gap is not enough the gap
    itself is widened, which widens the canvas by the same amount."""
    global GAP_CD, FIXED_W
    need = measure(None, None, COL_TITLE_D, COL_FS, "bold")
    # +0.004 in so the rounded gap cannot leave the title a hair too wide for wrap()
    GAP_CD = max(GAP_CD_BASE, round(need + COL_CLEAR - D_LABEL - D_W_FIX + 0.004, 4))
    FIXED_W = L_GUT + GAP_AB + GAP_BC + GAP_CD + D_LABEL + RIGHT
    return need, GAP_CD - GAP_CD_BASE


def d_title_lines():
    """d's column title, wrapped to the span it has.  After widen_gap_cd_for_d_title() this is one
    line; the wrap stays as a guard so the layout can never silently sit on panel c."""
    return wrap(None, None, COL_TITLE_D, d_title_span(), COL_FS, "bold")


def title_rows():
    return len(d_title_lines()) if COL_TITLES else 0


def letter_top():
    """Top of the panel-letter band: under the column titles when they are drawn."""
    return COL_TOP + (block_h(title_rows(), COL_FS) + COL_GAP if COL_TITLES else 0.0)


def row_top():
    """Top of the panels' own content: under the column titles and then the panel letters."""
    return letter_top() + LET_H + LET_GAP


COL_TITLE_AB = "BioReason reasoning trace accuracy"
COL_TITLE_C = "BioReason-Pro reasoning trace"
COL_TITLE_D = "Cell2Sentence-Scale cell type rationale"


FIXED_W = L_GUT + GAP_AB + GAP_BC + GAP_CD + D_LABEL + RIGHT
POOL = W - FIXED_W                   # the 7.00 in pool, kept for the record only

# filled in by allocate()
C_W = AX_W = M_H = M_W = CELL = D_W = C_X = None


def allocate(c_w):
    """Panel widths and the square cell pitch for a panel-c width.  a, b and d are pinned to the
    widths they had on the 7.00 in canvas; W grows or shrinks with panel c."""
    global C_W, AX_W, M_H, M_W, CELL, D_W, C_X, W
    C_W = c_w
    AX_W = AX_W_FIX
    D_W = D_W_FIX
    CELL = AX_W / rq3.NCOL_U
    M_H = CELL * len(rq3.PAIRS)
    M_W = CELL * (len(rq3.PAIRS) + 1)
    assert abs(M_W / 13 - M_H / 12) < 1e-9, (M_W / 13, M_H / 12)
    W = FIXED_W + 2 * AX_W + C_W + D_W
    C_X = L_GUT + 2 * AX_W + GAP_AB + GAP_BC
    assert abs(C_X + C_W + GAP_CD + D_LABEL + D_W + RIGHT - W) < 1e-9
    return CELL >= MIN_PITCH - 1e-9 and D_W >= D_W_MIN - 1e-9


YLABEL_AB = "True pair (ref>var), $n$ queries"
CBLABEL = "Proportion of responses"


def build(truth, data, P, secs):
    fig = plt.figure(figsize=(W, H))
    a_x = L_GUT
    b_x = a_x + AX_W + GAP_AB
    c_x = C_X
    d_lx = c_x + C_W + GAP_CD
    d_x0 = d_lx + D_LABEL            # where panel d would sit with no shift
    d_x = d_x0 - D_SHIFT             # panel d slides left to sit better under its wide title

    cmap = rq3.dna_cmap()
    cmap.set_bad(color="white", alpha=0.0)
    let_top = letter_top()
    r_top = row_top()
    mat_top = r_top + TIT
    col_titles = []
    if COL_TITLES:
        # All four column titles are FIGURE text anchored by their TOP at the same y, so they share
        # one baseline by construction; the assertion below measures it rather than trusting that.
        for x_c, lab in ((a_x + (2 * AX_W + GAP_AB) / 2, COL_TITLE_AB),
                         (c_x + C_W / 2, COL_TITLE_C),
                         (None, COL_TITLE_D)):
            if x_c is None:
                # d's title is wider than d; centre it on d's axes, slide it left only as far as
                # the right margin forces, and never let it reach panel c's block.
                lab = "\n".join(d_title_lines())
                half = max(measure(fig, None, q, COL_FS, "bold") for q in lab.split("\n")) / 2
                x_c = min(d_x0 + D_W / 2, W - RIGHT - half)   # NOT d_x: the title stays put
                x_c = max(x_c, c_x + C_W + COL_CLEAR + half)
            col_titles.append(txt(fig, x_c, COL_TOP, lab, COL_FS, ha="center",
                                  fontweight="bold", multialignment="center",
                                  linespacing=LS, zorder=5))
    im = None
    axes_ab = []
    for letter, ck, x0 in (("a", "rl", a_x), ("b", "sft", b_x)):
        cfg = rq3.CKPTS[ck]
        M, _ = rq3.matrix(truth, data[ck]["wt"])
        n_tot, diag = record_matrix(letter, ck, M, cfg)
        # DRAWN title only: the " (released)" suffix is stripped here and nowhere else, so the CSV
        # `arm` column and the standalone two-panel figures keep CKPTS[ck]["name"] verbatim.  The
        # condition (Evo2 intact) is in the caption, not in the title.
        title = (f"{cfg['name'].replace(' (released)', '')}\n"
                 f"Accuracy {diag}/{n_tot} = {diag / n_tot:.3f}")
        ax, im, empty = rq3.draw_matrix_panel(
            fig, x0, mat_top, AX_W, M_H, M, cmap=cmap, show_yticklabels=(letter == "a"),
            title=title, tick_fs=5.0, title_fs=5.6, label_fs=6.0, title_pad=2.5,
            ylabel=YLABEL_AB)
        assert not empty, empty
        axes_ab.append(ax)
        panel_letter(fig, x0 - (L_GUT if letter == "a" else 0.05), let_top, letter)
        print(f"  [{letter}] {cfg['name']} wt: diagonal {diag}/{n_tot} = {diag / n_tot:.4f}")

    # shared colour bar, horizontal, under a and b.  No note under it: the dotted diagonal outline
    # is described in the caption.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    low = min(min(t.get_window_extent(r).y0 for t in ax.get_xticklabels() if t.get_text())
              for ax in axes_ab)
    xlab_low = min(ax.xaxis.label.get_window_extent(r).y0 for ax in axes_ab)
    ab_bottom = H - min(low, xlab_low) / fig.dpi
    cb_y = ab_bottom + CB_TOP_GAP
    cb_x = a_x + (2 * AX_W + GAP_AB - CB_W) / 2
    cax = ax_in(fig, cb_x, cb_y, CB_W, CB_H)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1.0)
    cb.outline.set_linewidth(0.5)
    cax.set_xlabel(CBLABEL, fontsize=5.6, labelpad=1.5)
    fig.canvas.draw()
    ab_end = H - cax.xaxis.label.get_window_extent(r).y0 / fig.dpi

    # c
    panel_letter(fig, c_x, let_top, "c")
    c_end = draw_panel_c(fig, c_x, r_top, C_W, secs)

    # d: panel c of fig_rq3_rationale_panels, drawn by rationale_panels.panel_c
    # d's letter lines up with the LEFT EDGE of d's column title, which is wider than the panel
    # and starts well left of it; a, b and c keep their own columns' left edges.
    d_let_x = d_lx
    if col_titles:
        fig.canvas.draw()
        d_let_x = col_titles[2].get_window_extent(fig.canvas.get_renderer()).x0 / fig.dpi
        assert d_let_x >= c_x + C_W + 0.03, (d_let_x, c_x + C_W)
    panel_letter(fig, d_let_x, let_top, "d")
    ax_d = ax_in(fig, d_x, mat_top, D_W, M_H)
    rq2.ROWS.clear()
    rq2.panel_c(ax_d, P, rot=0)
    # Type size only: at ~1 in a matrix cannot carry a 7 pt x label, so a and b are at 6 pt, and
    # one axis-label size across the row is worth more than keeping the 7 pt house default in d
    # alone.  No bar, dot, interval or value label of the source panel is touched -- assert_panel_d
    # checks every drawn value against fig_rq3_rationale_panels_numbers.csv.
    ax_d.yaxis.label.set_fontsize(6.0)
    ax_d.tick_params(axis="y", labelsize=6.0)
    ax_d.yaxis.labelpad = D_LABELPAD       # 3.0 pt house default is more air than the label needs
    for row in rq2.ROWS:
        rec(panel="d", arm="C2S-Scale 27B", group=row.get("group", ""), x=row.get("x", ""),
            series=row["series"], value=row["value"], n=row["n"], ci_low=row.get("ci_low", ""),
            ci_high=row.get("ci_high", ""), drawn=True, source=row["source"], key=row["key"])
    fig.canvas.draw()
    # d's y label is two rotated lines deep; keep a real gap between it and panel c's last glyph
    d_lab_x = ax_d.yaxis.label.get_window_extent(r).x0 / fig.dpi
    assert d_lab_x >= c_x + C_W + 0.03, (d_lab_x, c_x + C_W)
    d_low = min(t.get_window_extent(r).y0 for t in ax_d.get_xticklabels() if t.get_text())
    d_bottom = H - d_low / fig.dpi
    h_d = [Patch(facecolor=col, edgecolor="black", linewidth=EDGE_LW, label=lab)
           for _, col, lab in rq2.ORDER]
    h_d.append(Patch(facecolor="white", edgecolor="black", linewidth=EDGE_LW, hatch=rq2.HATCH,
                     label="input\nshare"))
    # One row of three entries: a 2.00 in canvas has no room under d for a stacked legend.  Anchored
    # by its RIGHT edge to the figure's right margin, because at this width the row of three ends
    # within a thousandth of an inch of the canvas and a left anchor puts the last label over it.
    leg = fig.legend(handles=h_d, loc="upper right",
                     bbox_to_anchor=((W - RIGHT - D_SHIFT) / W, 1 - (d_bottom + 0.05) / H),
                     ncol=3,
                     frameon=False, fontsize=LEGEND_FS, handlelength=0.8, handletextpad=0.3,
                     borderaxespad=0.0, borderpad=0.0, labelspacing=0.3, columnspacing=0.35)
    fig.canvas.draw()
    lb = leg.get_window_extent(r)
    assert lb.x0 / fig.dpi >= c_x + C_W + 0.03, (lb.x0 / fig.dpi, c_x + C_W)
    d_end = H - lb.y0 / fig.dpi

    info = {}
    if col_titles:
        tops = [H - t.get_window_extent(r).y1 / fig.dpi for t in col_titles]
        spread = max(tops) - min(tops)
        assert spread <= 0.002, (spread, tops)
        boxes = [(t.get_window_extent(r).x0 / fig.dpi, t.get_window_extent(r).x1 / fig.dpi)
                 for t in col_titles]
        # the d title may run left over the c/d gap but must clear panel c's grey block
        # COL_CLEAR is reserved from the ADVANCE width; the rendered ink box is ~1 % wider
        # at the 100 dpi measuring canvas, so the realised clearance is a little less.
        assert boxes[2][0] >= c_x + C_W + 0.03, (boxes[2], c_x + C_W)
        info_lines = len(d_title_lines())
        assert boxes[2][1] <= W + 0.002, boxes[2]
        # and the a/b title must be centred on the PAIR of axes boxes
        ab_mid = a_x + (2 * AX_W + GAP_AB) / 2
        assert abs((boxes[0][0] + boxes[0][1]) / 2 - ab_mid) < 0.01
        ylab = axes_ab[0].yaxis.label.get_window_extent(r).x0 / fig.dpi
        ytk = min(t.get_window_extent(r).x0 for t in axes_ab[0].get_yticklabels()
                  if t.get_text()) / fig.dpi
        b_right = a_x + 2 * AX_W + GAP_AB
        info.update(ab_title_centre=(boxes[0][0] + boxes[0][1]) / 2,
                    ab_axes_centre=ab_mid,
                    ab_with_labels_centre=(min(ylab, ytk) + b_right) / 2,
                    ab_from_zero_centre=b_right / 2,
                    ab_left_ink=min(ylab, ytk), ab_right_frame=b_right)
        lets = [t for t in fig.texts if t.get_text() in ("a", "b", "c", "d")]
        let_top_m = min(H - t.get_window_extent(r).y1 / fig.dpi for t in lets)
        tit_bot = max(H - t.get_window_extent(r).y0 / fig.dpi for t in col_titles)
        info.update(title_tops=tops, d_title_rows=info_lines, d_letter_x=d_let_x,
                    d_letter_x_unshifted=d_lx, d_title_centre=(boxes[2][0] + boxes[2][1]) / 2, title_spread=spread, title_boxes=boxes,
                    letter_clearance=let_top_m - tit_bot, title_bottom=tit_bot,
                    letter_top_measured=let_top_m,
                    d_title_left_gap=boxes[2][0] - (c_x + C_W),
                    d_title_right_gap=W - boxes[2][1])
    fig.canvas.draw()
    d_lab_box = ax_d.yaxis.label.get_window_extent(r)
    info["d_label_gap"] = d_x - d_lab_box.x1 / fig.dpi
    info["d_label_clear_of_c"] = d_lab_box.x0 / fig.dpi - (c_x + C_W)
    lab_boxes = sorted((t.get_window_extent(r).x0 / fig.dpi, t.get_window_extent(r).x1 / fig.dpi)
                       for t in ax_d.texts if t.get_text() and abs(t.get_rotation()) < 1)
    info["d_value_labels"] = [t.get_text() for t in ax_d.texts if t.get_text()]
    info["d_label_min_gap"] = min(b[0] - a[1] for a, b in zip(lab_boxes, lab_boxes[1:]))
    info["d_bar_in"] = 0.26 / 2.0 * D_W
    if col_titles:
        info["d_overhang_left"] = d_x - boxes[2][0]
        info["d_overhang_right"] = boxes[2][1] - (d_x + D_W)
    info["d_axes_centre"] = d_x + D_W / 2
    info["d_axes_centre_unshifted"] = d_x0 + D_W / 2
    info["d_right_margin"] = W - (d_x + D_W)
    assert info["d_label_clear_of_c"] >= D_LAB_CLEAR_MIN - 0.002, info["d_label_clear_of_c"]
    return fig, dict(ab=ab_end, c=c_end, d=d_end, **info)


RATIONALE_CSV = os.path.join(FIG_DIR, "fig_rq3_rationale_panels_numbers.csv")
AB_CSV = {"a": os.path.join(FIG_DIR, "fig_rq3_trace_base_pair_numbers.csv"),
          "b": os.path.join(FIG_DIR, "fig_rq3_trace_base_pair_sft_numbers.csv")}


def assert_panel_ab(rows):
    """Panels a and b must reproduce, value for value, panel a of the two-panel figures they are
    lifted from.  Same series strings, so the comparison is exact rather than by eye."""
    for letter in ("a", "b"):
        path = AB_CSV[letter]
        if not os.path.exists(path):
            print(f"  [{letter}] {path} absent -- skipped")
            continue
        want = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["panel"] == "a":                      # its panel a == our panel {letter}
                    want[(r["series"], r["key"])] = float(r["value"])
        got = {(r["series"], r["key"]): float(r["value"])
               for r in rows if r["panel"] == letter}
        assert set(got) == set(want), (letter, sorted(set(got) ^ set(want))[:3])
        worst = max(abs(got[k] - want[k]) for k in got)
        assert worst < 1e-9, (letter, worst)
        print(f"  [{letter}] {len(got)} values match {os.path.basename(path)} panel a exactly "
              f"(max |difference| {worst:.2e})")


def assert_panel_d(rows):
    """Every panel d value must equal the value fig_rq3_rationale_panels_numbers.csv holds for its
    panel c."""
    want = {}
    with open(RATIONALE_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["panel"] == "c":
                want[(r["group"], r["series"], r["x"])] = float(r["value"])
    got = {(r["group"], r["series"], r["x"]): float(r["value"]) for r in rows}
    assert set(got) == set(want), (len(got), len(want), sorted(set(got) ^ set(want))[:3])
    worst = max(abs(got[k] - want[k]) for k in got)
    assert worst < 1e-9, worst
    print(f"  [d] {len(got)} values match fig_rq3_rationale_panels_numbers.csv panel c exactly "
          f"(max |difference| {worst:.2e})")


# ================================================================ QA
LINEWIDTH_IN = 5.50                  # \linewidth in this style
FLOOR_PT = 5.0                       # print-style floor for reproduced text


def scale():
    return LINEWIDTH_IN / W


def font_sizes(fig):
    """Every distinct font size actually drawn, with a count and one example."""
    fig.canvas.draw()
    items = list(fig.texts)
    for ax in fig.axes:
        items += list(ax.texts)
        items += [t for t in (ax.title, ax.xaxis.label, ax.yaxis.label) if t.get_text()]
        items += [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()]
        if ax.get_legend():
            items += list(ax.get_legend().get_texts())
    for leg in fig.legends:
        items += list(leg.get_texts())
    out = {}
    for t in items:
        if not t.get_text() or not t.get_visible():
            continue
        fs = round(float(t.get_fontsize()), 2)
        n, ex = out.get(fs, (0, ""))
        out[fs] = (n + 1, ex or t.get_text().replace("\n", " ")[:30])
    return dict(sorted(out.items(), reverse=True))


def check_panel_c_fits(fig):
    """Every text artist of panel c must sit inside its section rectangle.  check_overflow only
    looks at the canvas edge, so a wrapped line that runs past the grey block is invisible to it."""
    r = fig.canvas.get_renderer()
    bad = 0
    for t, (x0, x1) in PANEL_C_TEXTS:
        b = t.get_window_extent(r)
        if b.x0 / fig.dpi < x0 - 0.002 or b.x1 / fig.dpi > x1 + 0.002:
            bad += 1
            print(f"  PANEL-C OVERFLOW {t.get_text()[:34]!r}: "
                  f"[{b.x0 / fig.dpi:.3f}, {b.x1 / fig.dpi:.3f}] outside [{x0:.3f}, {x1:.3f}] in")
    print(f"[layout] {STEM}: panel c text outside its section rectangle: {bad} items")
    return bad


# ---------------------------------------------------------------- pixel-level checks
def _ink_runs(rgb, thresh=200):
    """Background runs between ink columns of one rendered strip: [(x_start, width), ...]."""
    ink = (rgb.sum(axis=2) < 3 * thresh).any(axis=0)
    cols = np.where(ink)[0]
    runs, s = [], None
    for x in range(cols.min(), cols.max() + 1):
        if not ink[x]:
            s = x if s is None else s
        elif s is not None:
            runs.append((s, x - s))
            s = None
    return runs, int(cols.min()), int(cols.max())


def check_highlight_gaps(png, last_line, last_segs, dpi=600, tol_px=3.0):
    """The highlighted phrase must be spaced exactly as one text artist would space it.

    trace_segments() checks the arithmetic; this reads pixels.  Two references are measured for the trace's last line: the same line drawn as
    ONE artist at the same size and dpi (what "no gap" means), and the median inter-word gap on
    that line.  The figure's own gap before the phrase must match the one-artist gap."""
    from PIL import Image

    def runs_of(arr, thresh=200):
        ink = (arr.sum(axis=2) < 3 * thresh).any(axis=0)
        c = np.where(ink)[0]
        out, st = [], None
        for x in range(c.min(), c.max() + 1):
            if not ink[x]:
                st = x if st is None else st
            elif st is not None:
                out.append((st - c.min(), x - st))
                st = None
        return out, int(c.min()), int(c.max())

    # ---- reference: the same line as ONE artist
    ref_png = os.path.join(os.path.dirname(png), "_line_ref.png")
    fw = measure(None, None, last_line, BODY_FS) + 0.2
    f = plt.figure(figsize=(fw, 0.14))
    f.text(0.05 / fw, 0.35, last_line, fontsize=BODY_FS, ha="left", va="baseline")
    f.savefig(ref_png, dpi=dpi, facecolor="white")
    plt.close(f)
    r_ref, _, _ = runs_of(np.array(Image.open(ref_png).convert("RGB")).astype(int))
    os.remove(ref_png)

    # ---- the figure itself: the line the highlight sits on
    a = np.array(Image.open(png).convert("RGB")).astype(int)
    tgt = np.array([int(COL_A[k:k + 2], 16) for k in (1, 3, 5)])
    ys, xs = np.where(np.abs(a - tgt).sum(axis=2) < 60)
    assert len(ys), "no highlight-coloured pixels in the PNG"
    keep = ys >= ys.max() - 14
    ys, xs = ys[keep], xs[keep]
    lo, hi = ys.min() - 7, ys.max() + 7
    x0 = int((C_X + SEC_PADX) * dpi) - 4
    x1 = int((C_X + C_W - SEC_PADX) * dpi) + 4
    r_fig, fig_x0, _ = runs_of(a[lo:hi + 1, x0:x1])

    pre = last_line[:last_line.index(PHRASE)]
    adv_px = measure(None, None, pre, BODY_FS) * dpi
    pick = lambda rr: min(rr, key=lambda t: abs(t[0] + t[1] - adv_px))
    g_fig, g_ref = pick(r_fig)[1], pick(r_ref)[1]
    n_sp = last_line.count(" ")
    words = sorted((w for _, w in r_ref), reverse=True)[:n_sp]
    med = float(np.median(words))
    after = [w for st, w in r_fig if abs(st + fig_x0 + x0 - xs.max()) <= 4]
    ok = abs(g_fig - g_ref) <= tol_px
    print(f"  [pixels] trace last line {last_line[-42:]!r} at {dpi} dpi")
    print(f"  [pixels]   gap before the phrase: figure {g_fig} px, one-artist reference {g_ref} px,"
          f" difference {abs(g_fig - g_ref)} px ({abs(g_fig - g_ref) / dpi:.4f} in)")
    print(f"  [pixels]   the {n_sp} inter-word gaps on that line (reference) {words}, "
          f"median {med:.1f} px")
    print(f"  [pixels]   gap between the phrase and the full stop {after[0] if after else None} px "
          f"(no space in the source, so this is a letter gap, not a word gap)")
    print(f"  [pixels]   segmented render matches the one-artist render within {tol_px:g} px: "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, (g_fig, g_ref)
    return dict(figure=g_fig, reference=g_ref, median_word=med, after=after[0] if after else None)


def check_ylabel_math(out_dir, dpi=600):
    """The italic n in a's y label must be spaced like a word, not like a math span.

    Rendered on its own strip so the gaps can be measured without the matrix behind it."""
    from PIL import Image
    p = os.path.join(out_dir, "_ylabel_probe.png")
    f = plt.figure(figsize=(2.2, 0.28))
    f.text(0.02, 0.35, YLABEL_AB, fontsize=6.0, ha="left", va="baseline")
    f.savefig(p, dpi=dpi, facecolor="white")
    plt.close(f)
    a = np.array(Image.open(p).convert("RGB")).astype(int)
    runs, _, _ = _ink_runs(a)
    space_px = 6.0 / 72 * 0.31833 * dpi
    words = sorted((w for _, w in runs if w >= 0.7 * space_px))
    assert len(words) >= 4, runs
    # the four word runs, in x order: True|pair, pair|(ref, ,|n, n|queries
    ordered = [w for _, w in runs if w >= 0.7 * space_px]
    med = float(np.median(ordered[:2]))             # the two gaps that are plainly word gaps
    before_n, after_n = ordered[-2], ordered[-1]
    print(f"  [pixels] y label, {dpi} dpi: word runs {ordered} px; median of the two plain word "
          f"gaps {med:.1f} px; run before $n$ {before_n} px; run after $n$ {after_n} px")
    os.remove(p)
    return dict(median=med, before=before_n, after=after_n, runs=ordered)


# ================================================================ caption
def write_tex(out_dir, truth, data, P, sizes):
    """The caption, with every N recomputed from the artifact the panel it describes reads."""
    v = {}
    for letter, ck in (("a", "rl"), ("b", "sft")):
        M, _ = rq3.matrix(truth, data[ck]["wt"])
        n = int(M.sum())
        diag = int(sum(M[i, i] for i in range(len(rq3.PAIRS))))
        col = M[:, :len(rq3.PAIRS)].sum(axis=0)
        j = int(np.argmax(col))
        v[letter] = dict(n=n, diag=diag, acc=diag / n, chance=rq3.chance_diagonal(M) / n,
                         none=int(M[:, -1].sum()), top=rq3.PAIRS[j], top_k=int(col[j]))
    rowN = [int(x) for x in rq3.matrix(truth, data["rl"]["wt"])[0].sum(axis=1)]
    pd = P["pooled"]["conditions"]
    g = lambda k, kk: 100 * pd[k][kk]["value"]
    f = lambda x: f"{x:.3f}"
    sc = scale()
    pt = ", ".join(f"{s:g}$\\to${s * sc:.2f}" for s in sizes)
    cap = rf"""% Built by reasoning_traces.py.  Include at \linewidth; the file is {W:.2f} x {H:.3f} in.
% \linewidth is {LINEWIDTH_IN:.2f} in in this style, so width=\linewidth scales the file by
% {sc:.4f}; drawn pt $\to$ printed pt: {pt}.
% Panel c width {C_W:.4f} in = {C_SCALE2:.2f} x {C_W_NARROW:.4f} in ({C_SCALE:.2f} x the
% {C_W_AT_H2:.3f} in that fits a 2.00 in canvas); a, b and d keep their 7.00 in widths, so the
% canvas width follows panel c.  At this width the figure needs {H:.3f} in of height (searched).
% Column titles: {COL_TITLE_AB!r} over a+b, {COL_TITLE_C!r} over c, {COL_TITLE_D!r} over d.
% Panel c carries OUR annotation at the foot of the output block, in the queried
% protein's blue {GO_NOTE_FG}: {GO_NOTE!r}
%   verified against {GO_CLOSURE_N} molecular function descendants of {DNA_BINDING_GO}
%   (is_a + part_of over go-basic.obo).
% Panel c colours: queried {QUERIED} {COL_B} ({wcag(COL_B, SEC_BG):.2f}:1 on {SEC_BG}), partner
% {PARTNER} {COL_A} ({wcag(COL_A, SEC_BG):.2f}:1), dE76 {delta_e76(COL_A, COL_B):.1f}.  The partner
% is OUTSIDE the protein hue -- a deliberate departure from the figure style, because no
% palette colour is both >= {MIN_CONTRAST}:1 on the section background and >= {MIN_DELTA_E:.0f} dE76 from the queried
% protein's {COL_B} (PROTEIN.perturbed, the semantically right role, is 2.23:1 and dE76 31.3).
% Numbers: {STEM}_numbers.csv (every drawn cell, every row total, every bar, and the panel c checks).
% a, b: ood_test of the pathway network split + the BioReason records of the RL and SFT checkpoints,
%       drawn by trace_base_pairs.draw_matrix_panel.
% c:    evidence_conflicts_rl/records.jsonl line {LINE_NO} ({PAIR_ID}, {COND}), verbatim.
% d:    rationale_order/rationale_order.json, drawn by rationale_panels.panel_c.
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/{STEM}.pdf}}
\caption{{\textbf{{Reasoning traces assert content the biological input does not determine.}}
\textbf{{(a, b)}} The reference-to-variant base pair BioReason states in its trace (columns, with a
\texttt{{none}} column for a trace that states no pair) against the query's true pair (rows), on the
${len(truth)}$ held-out queries (\texttt{{ood\_test}} of the pathway-network split, $145$ queries)
whose reference and variant windows differ at exactly one position; Evo2 is given the original DNA.
Each row is normalised by its own total, so a diagonal cell is the accuracy for that true pair; the
dotted outline marks the diagonal. Row
totals are in the $y$ tick labels of \textbf{{a}} and range from ${min(rowN)}$ to ${max(rowN)}$, and
\textbf{{b}} shares them. The stated pair equals the true pair on ${v['a']['diag']}/{v['a']['n']}=
{f(v['a']['acc'])}$ for the released RL checkpoint and ${v['b']['diag']}/{v['b']['n']}={f(v['b']['acc'])}$
for the released SFT checkpoint, against ${f(v['a']['chance'])}$ and ${f(v['b']['chance'])}$ expected if
the stated pair were independent of the true pair. RL states \texttt{{{v['a']['top']}}} on
${v['a']['top_k']}$ of ${v['a']['n']}$ queries and never states no pair; SFT states no pair on
${v['b']['none']}$. \textbf{{(c)}} One BioReason-Pro evidence conflict, verbatim from the stored
record: the ESM3 representation and the GO-GPT terms come from the queried protein Q62227, the
InterPro domains from the partner Q505F1, and the emitted GO terms are shown as the model wrote them.
\textbf{{(d)}} Share of the genes C2S-Scale 27B references that are ribosomal-protein genes, and DEG
precision, pooled over $2{{,}}000$ cells from five atlases ($400$ each), with the cell sentence in
descending-expression order (${g('wt', 'ribosomal_share'):.0f}$ and ${g('wt', 'deg_precision'):.0f}$
per cent) and with the same genes in a random order (${g('scramble_rank', 'ribosomal_share'):.0f}$ and
${g('scramble_rank', 'deg_precision'):.0f}$), beside the share among the input genes
(${g('wt', 'ribosomal_chance'):.0f}$ and ${g('wt', 'deg_chance'):.0f}$). Dots are the five atlases;
error bars are 95\% cell-bootstrap intervals.}}
\label{{fig:rq3_traces}}
\end{{figure}}
"""
    open(os.path.join(out_dir, f"{STEM}.tex"), "w").write(cap)
    print(f"  wrote {os.path.join(out_dir, f'{STEM}.tex')}")


# ================================================================ the panel-c width search
def c_height(c_w):
    """Panel c's height in inches at a given panel width, without building the figure."""
    scratch = plt.figure(figsize=(W, H))
    scratch.canvas.draw()
    secs, total, n_lines = panel_c_lines(scratch, scratch.canvas.get_renderer(),
                                         c_w - 2 * SEC_PADX, verbose=False)
    plt.close(scratch)
    return secs, total, n_lines


def _qa(fig):
    """The three QA counts, quietly."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_fit = check_panel_c_fits(fig)
        n_over = len(check_overflow(fig, STEM))
        n_coll = check_text_collisions(fig, STEM)
    return n_fit, n_over, n_coll, buf.getvalue()


def trial(truth, data, P, c_rows, h):
    """Build the whole figure at canvas height `h` and return the three QA counts and the ends."""
    set_h(h)
    secs, total, n_lines = c_height(C_W)
    del ROWS[:]
    for row in c_rows:
        rec(**row)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fig, ends = build(truth, data, P, secs)
    n_fit, n_over, n_coll, qa_log = _qa(fig)
    need = max(ends["ab"], ends["c"], ends["d"]) + BOT_PAD
    ok = n_fit == 0 and n_over == 0 and n_coll == 0 and need <= h + 1e-9
    return dict(h=h, ok=ok, counts=(n_fit, n_over, n_coll), need=need, ends=ends,
                c_h=total, lines=n_lines, fig=fig, secs=secs, log=buf.getvalue() + qa_log)


def measure_d_shift(truth, data, P, c_rows, h=2.60):
    """How far panel d's axes can slide left before its rotated y axis label reaches panel c.

    Measured on a throwaway build at D_SHIFT = 0 rather than assumed from the gutter arithmetic,
    then frozen, so every later build (the height search included) uses the same geometry."""
    global D_SHIFT
    D_SHIFT = 0.0
    t = trial(truth, data, P, c_rows, h)
    plt.close(t["fig"])
    clear0 = t["ends"]["d_label_clear_of_c"]
    off0 = t["ends"]["d_axes_centre"] - t["ends"]["d_title_centre"]
    # 0.001 in short of the floor: the guard reads the hinted extent and the shift is
    # derived from the same number, so landing exactly on it fails by rounding.
    D_SHIFT = max(0.0, round(clear0 - D_LAB_CLEAR_MIN - 0.001, 3))
    print(f"== panel d shift: its y axis label clears panel c by {clear0:.4f} in with no shift; at "
          f"a {D_LAB_CLEAR_MIN:.3f} in floor the axes can move {D_SHIFT:.4f} in left")
    return clear0, off0, D_SHIFT


def search_h(truth, data, P, c_rows, h0=2.00, step=0.005, hi=3.20):
    """The SMALLEST canvas height at which the whole figure fits with zero QA items.

    Ascending grid from h0, building the real figure and running check_panel_c_fits,
    check_overflow and check_text_collisions at every step.  Returns (smallest pass, largest fail,
    the whole trace of the search)."""
    print(f"== canvas height search at W = {W:.4f} in, panel c = {C_W:.4f} in: ascending grid "
          f"from {h0:.3f} in, step {step:g} in")
    trace, last_fail, first_pass = [], None, None
    h = h0
    while h <= hi + 1e-9:
        t = trial(truth, data, P, c_rows, round(h, 4))
        plt.close(t["fig"])
        trace.append((t["h"], t["ok"], t["counts"], round(t["need"], 4), round(t["c_h"], 4)))
        flag = "PASS" if t["ok"] else "FAIL"
        extra = "" if t["ok"] else f" (needs {t['need']:.3f} in, over by {t['need'] - t['h']:.3f})"
        if t["ok"] or h == h0 or abs(h - h0) < 0.051 or t["ok"]:
            print(f"   H {t['h']:.3f}  c_h {t['c_h']:.3f}  lines {t['lines']}  "
                  f"checks (fit {t['counts'][0]}, overflow {t['counts'][1]}, "
                  f"collisions {t['counts'][2]})  -> {flag}{extra}")
        if t["ok"]:
            first_pass = t
            break
        last_fail = t
        h += step
    assert first_pass is not None, trace
    if last_fail is not None:
        print(f"   largest failing height {last_fail['h']:.3f} in: needs {last_fail['need']:.3f} "
              f"in, checks {last_fail['counts']}")
    return first_pass, last_fail, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=FIG_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    init_print_style()
    matplotlib.rcParams["hatch.linewidth"] = 0.45
    # Pin the mathtext families to the body font so the italic n in a's y label is the same
    # typeface as the rest of the label rather than a Computer-Modern-style math italic.
    matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "DejaVu Sans",
                                "mathtext.it": "DejaVu Sans:italic",
                                "mathtext.bf": "DejaVu Sans:bold", "mathtext.default": "it"})

    print(f"== fixed gutters {FIXED_W:.3f} in; a, b axes pinned at {AX_W_FIX:.4f} in each and "
          f"d at {D_W_FIX:.4f} in, so the canvas width follows panel c")
    print(f"== panel c colours on {SEC_BG}:")
    for nm, c in (("queried  " + QUERIED, COL_B), ("partner  " + PARTNER, COL_A),
                  ("PROTEIN.perturbed (rejected)", PROTEIN.perturbed),
                  ("PROTEIN.model     (rejected)", PROTEIN.model),
                  ("DNA.model_dark    (rejected)", DNA.model_dark),
                  ("CTRL[2]           (rejected)", CTRL[2])):
        print(f"     {nm:30s} {c}  contrast {wcag(c, SEC_BG):5.2f}:1  "
              f"dE76 vs {COL_B} {delta_e76(c, COL_B):5.1f}")

    print("== panel c: verifying every string against the record")
    _, c_rows = verify_panel_c()
    print("== panels a and b: loading the held-out traces")
    truth, data, states_all = rq3.load()
    print("== panel d: loading rationale_order.json")
    P = rq2.load_position()

    need_d, extra = widen_gap_cd_for_d_title()
    print(f"== d's column title is {need_d:.4f} in at {COL_FS:g} pt bold; the c/d gap goes "
          f"{GAP_CD_BASE:.3f} -> {GAP_CD:.4f} in (+{extra:.4f} in of canvas width) so it stays on "
          f"one line with {COL_CLEAR:.3f} in of clearance from panel c")

    print("== panel c: GO:0003677 closure check")
    ids = sorted(g for _, terms in EMITTED for g, _ in terms)
    go_info = verify_go_absent(ids)

    # ---- the 20 % narrower panel c, and what canvas height it needs ----
    allocate(C_W_NARROW)
    _, h_ref, lines_ref = c_height(C_W_NARROW)
    allocate(C_W_TARGET)
    _, h_new, lines_new = c_height(C_W_TARGET)
    print(f"== panel c {C_W_NARROW:.4f} in x {C_SCALE2:.2f} = {C_W:.4f} in "
          f"(text box {C_W - 2 * SEC_PADX:.4f} in); canvas width {W:.4f} in")
    print(f"   panel c lines (inputs, trace, output GO + our line): {lines_ref} at "
          f"{C_W_NARROW:.4f} in -> {lines_new} at {C_W:.4f} in; height {h_ref:.3f} -> "
          f"{h_new:.3f} in")


    clear0, off0, shift = measure_d_shift(truth, data, P, c_rows)
    best, worst, _ = search_h(truth, data, P, c_rows)
    # what the column titles alone cost: the same search with them switched off
    global COL_TITLES
    COL_TITLES = False
    best_nt, _, _ = search_h(truth, data, P, c_rows)
    COL_TITLES = True
    print(f"== minimum canvas height {best['h']:.3f} in; without the column titles "
          f"{best_nt['h']:.3f} in, so the titles cost {best['h'] - best_nt['h']:.3f} in and the "
          f"narrower panel c costs {best_nt['h'] - 2.00:.3f} in of the "
          f"{best['h'] - 2.00:.3f} in total increase over 2.00 in")

    set_h(best["h"])
    print(f"== geometry: canvas {W:.2f} x {H:.3f} in; a, b axes {AX_W:.4f} x {M_H:.4f} in, "
          f"cell {CELL:.6f} in (M_W/13 = {M_W / 13:.6f}, M_H/12 = {M_H / 12:.6f}); "
          f"c {C_W:.4f} in; d {D_W:.4f} in")

    secs, total, n_lines = c_height(C_W)
    del ROWS[:]
    for row in c_rows:
        rec(**row)
    fig, ends = build(truth, data, P, secs)
    need = max(ends["ab"], ends["c"], ends["d"]) + BOT_PAD
    print(f"  [layout] a/b end {ends['ab']:.3f}, c end {ends['c']:.3f}, d end {ends['d']:.3f} in; "
          f"with {BOT_PAD:.2f} in bottom pad the column needs {need:.3f} of {H:.3f} in")
    print(f"  [titles] tops {[round(v, 4) for v in ends['title_tops']]} in, spread "
          f"{ends['title_spread']:.5f} in (<= 0.002 asserted); titles occupy "
          f"{COL_TOP:.3f}-{ends['title_bottom']:.3f} in, panel letters start at "
          f"{ends['letter_top_measured']:.3f} in, clearance {ends['letter_clearance']:.3f} in")
    print(f"  [anchors] a/b title centre {ends['ab_title_centre']:.4f} in | two axes boxes "
          f"(a's left frame to b's right frame) {ends['ab_axes_centre']:.4f} | column including "
          f"a's y tick labels and y axis label (left ink {ends['ab_left_ink']:.4f}) "
          f"{ends['ab_with_labels_centre']:.4f} | from x = 0 to b's right frame "
          f"({ends['ab_right_frame']:.4f}) {ends['ab_from_zero_centre']:.4f}")
    print(f"  [titles] d's title is on {ends['d_title_rows']} line(s); it clears panel c by "
          f"{ends['d_title_left_gap']:.3f} in and the right canvas edge by "
          f"{ends['d_title_right_gap']:.3f} in")
    print(f"  [panel d] letter x {ends['d_letter_x_unshifted']:.4f} -> {ends['d_letter_x']:.4f} in "
          f"(d's title left edge); clears panel c's block by "
          f"{ends['d_letter_x'] - (C_X + C_W):.4f} in")
    print(f"  [panel d] axes shifted left {D_SHIFT:.4f} in; y label clears panel c "
          f"{clear0:.4f} -> {ends['d_label_clear_of_c']:.4f} in (floor {D_LAB_CLEAR_MIN:.3f}, "
          f"asserted); right margin {RIGHT:.4f} -> {ends['d_right_margin']:.4f} in")
    print(f"  [panel d] axes centre {ends['d_axes_centre_unshifted']:.4f} -> "
          f"{ends['d_axes_centre']:.4f} in against a title centre of "
          f"{ends['d_title_centre']:.4f} in; offset {off0:+.4f} -> "
          f"{ends['d_axes_centre'] - ends['d_title_centre']:+.4f} in")
    print(f"  [panel d] width {D_W:.4f} in; bar {ends['d_bar_in']:.4f} in (floor 0.030); value "
          f"labels {ends['d_value_labels']} with a minimum gap of {ends['d_label_min_gap']:.4f} in")
    print(f"  [panel d] its title overhangs the axes by {ends['d_overhang_left']:.4f} in on the "
          f"left and {ends['d_overhang_right']:.4f} in on the right")
    print(f"  [panel d] y label gap to the axes {ends['d_label_gap']:.4f} in "
          f"(labelpad {D_LABELPAD} pt, gutter {D_LABEL:.3f} in); label clears panel c by "
          f"{ends['d_label_clear_of_c']:.4f} in")
    if need > H:
        raise SystemExit(f"STOP: the panels need {need:.3f} in but the canvas is "
                         f"{H:.3f} in -- short by {need - H:.3f} in.")

    assert_panel_ab(ROWS)
    assert_panel_d([r for r in ROWS if r["panel"] == "d"])
    n_wide = check_panel_c_fits(fig)
    n_over = len(check_overflow(fig, STEM))
    n_coll = check_text_collisions(fig, STEM)
    print(f"[layout] {STEM}: check_overflow {n_over} items, check_text_collisions {n_coll} items, "
          f"panel-c overflow {n_wide} items")

    sizes = font_sizes(fig)
    sc = scale()
    print(f"[type] drawn at {W:.2f} in, included at width=\\linewidth ({LINEWIDTH_IN:.2f} in) "
          f"-> scale {sc:.4f}")
    print(f"  {'drawn pt':>9}  {'printed pt':>10}  {'n':>4}  below {FLOOR_PT:.1f} pt?  example")
    for fs, (n, ex) in sizes.items():
        pr = fs * sc
        print(f"  {fs:9.2f}  {pr:10.2f}  {n:4d}  {'YES' if pr < FLOOR_PT else 'no ':>13}  {ex!r}")
    under = [fs for fs in sizes if fs * sc < FLOOR_PT]
    if under:
        print(f"[type] WARNING: {len(under)} of {len(sizes)} drawn sizes print below "
              f"{FLOOR_PT:.1f} pt at width=\\linewidth: "
              + ", ".join(f"{fs:g} pt -> {fs * sc:.2f} pt" for fs in under))

    emit(fig, STEM, a.out_dir)
    write_tex(a.out_dir, truth, data, P, list(sizes))

    print("== pixel checks on the rendered PNG")
    body = [sec for sec, _ in secs if "segments" in sec][0]
    check_highlight_gaps(os.path.join(a.out_dir, f"{STEM}.png"), body["body"][-1],
                         body["segments"][-1])
    check_ylabel_math(a.out_dir)

    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_KEYS, extrasaction="ignore")
        w.writeheader()
        for row in ROWS:
            w.writerow({k: row.get(k, "") for k in CSV_KEYS})
    n_by = {}
    for row in ROWS:
        n_by[row["panel"]] = n_by.get(row["panel"], 0) + 1
    print(f"wrote {p} ({len(ROWS)} rows; " +
          ", ".join(f"panel {k} {v}" for k, v in sorted(n_by.items())) + ")")


if __name__ == "__main__":
    main()
