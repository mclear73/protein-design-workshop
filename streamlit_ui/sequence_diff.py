"""
Sequence-diff view for the Step 3 results pane.

Renders three components inline below the structure viewer:

  1. Headline metrics — % identity, mutation count, anchors-preserved tally.
  2. Aligned sequence diff — WT on top, designed below, wrapped at `wrap`
     residues per block, position ruler + anchor-marker row above each block.
  3. Caption — lane-specific narrative explaining what the participant is
     looking at (passed in from app.py's TARGETS[team]["diff_caption"]).

Design constraints (from sequence_diff_workplan.md):
  - ProteinMPNN fixbb preserves sequence length. If lengths differ, raise
    rather than paper over with a pairwise alignment.
  - Single render call per Step 3 page. No modals, no separate routes.
  - Dark-mode-friendly colors; inline styles only (Streamlit strips
    <style> blocks).
  - Resist scope creep: this is a 1-screen view answering "how much did
    ProteinMPNN actually change?" Nothing else.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st


# =============================================================================
# Color tokens — chosen to read on dark mode (the workshop's default theme)
# =============================================================================
_DIM            = "#7a8a99"  # conserved residue (eye should ignore)
_MUT_FG         = "#ffd166"  # mutated residue in the designed row (warm yellow)
_ANCHOR_OK_BG   = "rgba(46, 204, 113, 0.18)"  # soft green tint
_ANCHOR_FAIL_BG = "rgba(231, 76, 60, 0.28)"   # red tint — louder than the OK
_RULER_FG       = "#5d6d7e"  # position ruler

# Marker characters in the row above the WT sequence.
_ANCHOR_GLYPH  = "▼"   # must-preserve catalytic anchors
_SUPPORT_GLYPH = "•"   # other functional residues (pocket, oxyanion, etc.)


# =============================================================================
# Data shapes
# =============================================================================
@dataclass
class DiffResult:
    """Position-by-position diff between WT and designed sequence."""
    wt_seq: str
    designed_seq: str
    length: int
    n_mutations: int
    pct_identity: float                       # 0.0–100.0
    mutation_positions: list[int]             # 1-indexed
    mutations_by_position: dict[int, tuple[str, str]]  # {pos: (wt_aa, des_aa)}


@dataclass
class AnchorReport:
    """How the lane's functional residues fared under the design."""
    total_anchors: int                        # count of "must-preserve" (anchor) residues in range
    preserved: list[dict]                     # anchors whose AA matched
    mutated: list[tuple[dict, str]]           # (anchor, designed_aa)
    catalytic_preserved: bool                 # True iff every anchor in the must-preserve allowlist is preserved
    n_supporting_preserved: int               # functional residues outside the allowlist that were preserved
    n_supporting_mutated: int                 # ... that were mutated


# =============================================================================
# Pure helpers — no Streamlit
# =============================================================================
def compute_diff(wt_seq: str, designed_seq: str) -> DiffResult:
    """
    Position-by-position diff. ProteinMPNN fixbb preserves length, so any
    length mismatch is a data bug (wrong reference loaded, or designs taken
    from the wrong target). Raise loudly rather than silently truncating.
    """
    if len(wt_seq) != len(designed_seq):
        raise ValueError(
            f"WT and designed sequences have different lengths "
            f"({len(wt_seq)} vs {len(designed_seq)}). "
            "ProteinMPNN fixbb preserves length — a mismatch usually means "
            "the wrong reference sequence is loaded, or the design output "
            "belongs to a different target than the one currently displayed."
        )

    n = len(wt_seq)
    muts_by_pos: dict[int, tuple[str, str]] = {}
    mut_positions: list[int] = []
    for i, (w, d) in enumerate(zip(wt_seq, designed_seq), start=1):
        if w != d:
            muts_by_pos[i] = (w, d)
            mut_positions.append(i)

    n_mut = len(mut_positions)
    pct_id = 100.0 * (n - n_mut) / n if n else 0.0

    return DiffResult(
        wt_seq=wt_seq,
        designed_seq=designed_seq,
        length=n,
        n_mutations=n_mut,
        pct_identity=pct_id,
        mutation_positions=mut_positions,
        mutations_by_position=muts_by_pos,
    )


def check_anchors(diff: DiffResult,
                  anchors: list[dict],
                  catalytic_categories: list[str]) -> AnchorReport:
    """
    Classify each functional residue (the in-range ones from
    `_resolve_functional_residues`) as preserved or mutated, and split into
    the must-preserve "anchor" set (categories in `catalytic_categories`)
    vs. the supporting cast.

    Out-of-range annotations are simply ignored — they don't appear in the
    diff sequence so they can't be checked. That matches how the conservation
    plot handles them already.
    """
    in_range = [a for a in anchors if a.get("in_range")]

    must_preserve = [a for a in in_range if a["category"] in catalytic_categories]
    supporting    = [a for a in in_range if a["category"] not in catalytic_categories]

    preserved: list[dict] = []
    mutated: list[tuple[dict, str]] = []
    for a in must_preserve:
        pos = a["position_msa"]
        if pos in diff.mutations_by_position:
            _wt, designed_aa = diff.mutations_by_position[pos]
            mutated.append((a, designed_aa))
        else:
            preserved.append(a)

    n_sup_pres = sum(
        1 for a in supporting if a["position_msa"] not in diff.mutations_by_position
    )
    n_sup_mut = sum(
        1 for a in supporting if a["position_msa"] in diff.mutations_by_position
    )

    return AnchorReport(
        total_anchors=len(must_preserve),
        preserved=preserved,
        mutated=mutated,
        catalytic_preserved=(len(mutated) == 0 and len(must_preserve) > 0),
        n_supporting_preserved=n_sup_pres,
        n_supporting_mutated=n_sup_mut,
    )


# =============================================================================
# Streamlit renderer
# =============================================================================
def render_diff_view(diff: DiffResult,
                     anchor_report: AnchorReport,
                     anchors: list[dict],
                     catalytic_categories: list[str],
                     category_colors: dict,
                     lane_caption: str = "",
                     wrap: int = 60) -> None:
    """
    Render the three-component diff view inline in the current Streamlit
    container. `anchors` is the lane's normalized functional-residue list
    (output of _resolve_functional_residues — list of dicts with position_msa,
    label, category, etc.). `category_colors` is the lane's color palette
    keyed by category name.
    """
    # ─── Metrics strip ────────────────────────────────────────────────────
    cols = st.columns(3)
    cols[0].metric(
        "Identity to WT",
        f"{diff.pct_identity:.1f}%",
        help="Percentage of positions where the designed sequence matches "
             "the wild-type. ProteinMPNN typically rewrites 40–60% of a "
             "well-folded target while keeping the backbone identical.",
    )
    cols[1].metric(
        "Mutations",
        f"{diff.n_mutations:,}",
        delta=f"of {diff.length:,} positions",
        delta_color="off",
        help="Number of residues changed between WT and designed.",
    )
    if anchor_report.total_anchors > 0:
        n_preserved = anchor_report.total_anchors - len(anchor_report.mutated)
        if anchor_report.catalytic_preserved:
            cols[2].metric(
                "Catalytic anchors held",
                f"{n_preserved} / {anchor_report.total_anchors}",
                delta="✓ all preserved",
                delta_color="normal",
                help="The lane's must-preserve catalytic residues. "
                     "If all are held, the design has a chance at activity.",
            )
        else:
            cols[2].metric(
                "Catalytic anchors held",
                f"{n_preserved} / {anchor_report.total_anchors}",
                delta=f"⚠ {len(anchor_report.mutated)} mutated",
                delta_color="inverse",
                help="The lane's must-preserve catalytic residues. "
                     "Any mutation here means the design has likely lost activity.",
            )
    else:
        cols[2].metric("Catalytic anchors held", "—",
                       help="No anchors in range for this construct.")

    # ─── Anchor-mutation callout (loud when it fires) ─────────────────────
    if anchor_report.mutated:
        bullets = "\n".join(
            f"- **{a['label']}** ({a['category']}): "
            f"WT **{a['expected_aa']}** → designed **{des_aa}**"
            for a, des_aa in anchor_report.mutated
        )
        st.error(
            "**Catalytic anchor(s) mutated.** The design is unlikely to "
            "retain activity even if AF2 predicts the same fold:\n\n" + bullets
        )

    # ─── The aligned sequence diff ────────────────────────────────────────
    must_preserve_pos = {a["position_msa"]: a for a in anchors
                         if a.get("in_range") and a["category"] in catalytic_categories}
    supporting_pos = {a["position_msa"]: a for a in anchors
                      if a.get("in_range") and a["category"] not in catalytic_categories}

    html_blocks = []
    for start in range(0, diff.length, wrap):
        end = min(start + wrap, diff.length)
        html_blocks.append(_render_block(
            wt=diff.wt_seq[start:end],
            des=diff.designed_seq[start:end],
            start_pos=start + 1,           # 1-indexed
            block_len=end - start,
            must_preserve_pos=must_preserve_pos,
            supporting_pos=supporting_pos,
            category_colors=category_colors,
        ))

    pre_style = (
        "background: rgba(255,255,255,0.02); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 6px; padding: 12px 14px; "
        "font-family: ui-monospace, 'SF Mono', Menlo, Monaco, Consolas, "
        "'Liberation Mono', 'Courier New', monospace; "
        "font-size: 13px; line-height: 1.45; "
        f"color: {_DIM}; "
        "white-space: pre; overflow-x: auto;"
    )
    st.markdown(
        f'<pre style="{pre_style}">' + "\n\n".join(html_blocks) + "</pre>",
        unsafe_allow_html=True,
    )

    # ─── Legend ───────────────────────────────────────────────────────────
    legend_bits = [
        f'<span style="color:{_DIM}">A</span> conserved',
        f'<span style="color:{_MUT_FG};font-weight:700">A</span> mutated',
        f'<span style="background:{_ANCHOR_OK_BG};padding:0 4px;border-radius:3px">A</span> anchor preserved',
        f'<span style="background:{_ANCHOR_FAIL_BG};padding:0 4px;border-radius:3px">A</span> anchor mutated',
        f'<span style="color:{_RULER_FG}">{_ANCHOR_GLYPH}</span> catalytic anchor',
        f'<span style="color:{_RULER_FG}">{_SUPPORT_GLYPH}</span> supporting residue (pocket / oxyanion / etc.)',
    ]
    st.markdown(
        '<div style="font-size:12px;color:#9aa5b1;margin-top:6px;">' +
        '&nbsp;&nbsp;·&nbsp;&nbsp;'.join(legend_bits) +
        '</div>',
        unsafe_allow_html=True,
    )

    # ─── Lane-specific narrative caption ─────────────────────────────────
    if lane_caption:
        st.caption(lane_caption)


# =============================================================================
# Block rendering — internal
# =============================================================================
# Left-side row labels. 5 chars + ":" so the colon aligns at column 5.
# Keep these short — they're decorative.
_LABEL_WIDTH = 5
_LABEL_RULER = "pos".ljust(_LABEL_WIDTH) + " "
_LABEL_ANN   = "ann".ljust(_LABEL_WIDTH) + " "
_LABEL_WT    = "WT".ljust(_LABEL_WIDTH)  + " "
_LABEL_DES   = "des".ljust(_LABEL_WIDTH) + " "


def _render_block(*, wt: str, des: str, start_pos: int, block_len: int,
                  must_preserve_pos: dict, supporting_pos: dict,
                  category_colors: dict) -> str:
    """
    Build the HTML for one wrapped block: ruler, annotation row, WT row,
    designed row. Returns the four <span>-laden lines joined by '\n'.
    """
    # --- Ruler row: every 10th position, right-justified in a 10-char cell.
    # Build as a single string, then color it via one outer span.
    ruler = []
    for i in range(block_len):
        pos = start_pos + i
        # The number ends at every multiple of 10. So we lay out "         10"
        # (9 spaces + "10") across positions 1..10; "         20" across
        # 11..20; etc. — i.e., the number's last digit sits at the multiple-of-10
        # column.
        if pos % 10 == 0:
            num = str(pos)
            # Walk back and overwrite the previous (len(num)-1) ruler chars
            for j, ch in enumerate(num):
                idx = i - (len(num) - 1 - j)
                if 0 <= idx < len(ruler):
                    ruler[idx] = ch
                else:
                    ruler.append(ch)
            # Ensure ruler ends right at column i — it may already
            if len(ruler) <= i:
                ruler.append(num[-1])
        else:
            if len(ruler) <= i:
                ruler.append(" ")
    # Pad to block_len in case the loop didn't reach
    while len(ruler) < block_len:
        ruler.append(" ")
    ruler_str = "".join(ruler[:block_len])
    ruler_line = (
        _LABEL_RULER +
        f'<span style="color:{_RULER_FG}">{html.escape(ruler_str)}</span>'
    )

    # --- Annotation row: a glyph above each annotated position.
    ann_parts = [_LABEL_ANN]
    for i in range(block_len):
        pos = start_pos + i
        if pos in must_preserve_pos:
            a = must_preserve_pos[pos]
            color = category_colors.get(a["category"], "#888")
            tooltip = html.escape(f'{a["label"]} ({a["category"]}) — must-preserve anchor')
            ann_parts.append(
                f'<span title="{tooltip}" style="color:{color};font-weight:700">'
                f'{_ANCHOR_GLYPH}</span>'
            )
        elif pos in supporting_pos:
            a = supporting_pos[pos]
            color = category_colors.get(a["category"], "#888")
            tooltip = html.escape(f'{a["label"]} ({a["category"]}) — supporting residue')
            ann_parts.append(
                f'<span title="{tooltip}" style="color:{color}">{_SUPPORT_GLYPH}</span>'
            )
        else:
            ann_parts.append(" ")
    ann_line = "".join(ann_parts)

    # --- WT row: dim. Anchor positions get a soft green/red background
    # (green if preserved, red if mutated).
    wt_parts = [_LABEL_WT]
    for i in range(block_len):
        pos = start_pos + i
        wt_aa = wt[i]
        des_aa = des[i]
        is_mut = wt_aa != des_aa
        anchor = must_preserve_pos.get(pos) or supporting_pos.get(pos)
        ch = html.escape(wt_aa)
        if anchor is not None:
            # Soft tint behind the anchor's WT char too (legend symmetry)
            bg = _ANCHOR_FAIL_BG if (is_mut and pos in must_preserve_pos) else _ANCHOR_OK_BG
            wt_parts.append(
                f'<span style="background:{bg};color:{_DIM}">{ch}</span>'
            )
        else:
            # Plain dim — no inline style needed (the pre's color is _DIM)
            wt_parts.append(ch)
    wt_line = "".join(wt_parts)

    # --- Designed row: warm-yellow bold for mutations, anchor backgrounds
    # on anchored positions.
    des_parts = [_LABEL_DES]
    for i in range(block_len):
        pos = start_pos + i
        wt_aa = wt[i]
        des_aa = des[i]
        is_mut = wt_aa != des_aa
        anchor = must_preserve_pos.get(pos) or supporting_pos.get(pos)
        ch = html.escape(des_aa)
        # Compute the foreground first
        if is_mut:
            fg_style = f"color:{_MUT_FG};font-weight:700"
        else:
            fg_style = f"color:{_DIM}"
        if anchor is not None:
            bg = _ANCHOR_FAIL_BG if (is_mut and pos in must_preserve_pos) else _ANCHOR_OK_BG
            des_parts.append(f'<span style="{fg_style};background:{bg}">{ch}</span>')
        elif is_mut:
            des_parts.append(f'<span style="{fg_style}">{ch}</span>')
        else:
            # Plain dim — inherits from pre
            des_parts.append(ch)
    des_line = "".join(des_parts)

    return "\n".join([ruler_line, ann_line, wt_line, des_line])
