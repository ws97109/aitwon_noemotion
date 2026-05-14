"""Fix figure-vs-text overlap issues in 論文＿李昇峰_v18.docx.

Findings (from scan):
  • One floating element (a cover-page text box, 文字方塊 27) uses
    `<wp:wrapNone/>` and `behindDoc="0"`.  With wrap = none the textbox does
    not push body text out of its area; with behindDoc = 0 the textbox is
    drawn IN FRONT of the body text — so any body text underneath gets
    visually covered.  Fix: switch to `<wp:wrapTopAndBottom/>` so subsequent
    text starts below the textbox instead of behind it; also set
    `behindDoc="1"` as a safety net so even any near-by run sits on top.
  • All 15 body images are inline.  Inline images do not float, so the only
    overlap they can produce is when Word splits the image paragraph from
    its caption — half of the figure ends up on one page, the caption on
    the next, leaving a tall blank that visually looks like overlap.
    Fix: set `keepNext` on every figure paragraph so the caption stays with
    the image; set `keepLines` on the caption so it doesn't split.

Output:  論文＿李昇峰_v18_fixed.docx  (next to the original at repo root)

Run:  python3 docs/fix_v18_figure_overlap.py
"""
import os, shutil
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree


SRC = Path("/Users/lishengfeng/Desktop/aitwon_noemotion/論文＿李昇峰_v18.docx")
OUT = Path("/Users/lishengfeng/Desktop/aitwon_noemotion/論文＿李昇峰_v18_fixed.docx")

WP_NS  = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
WP14_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
W_NS   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def set_or_get(parent, tag, ns=W_NS):
    """Ensure `parent` has a single child element of tag `tag` in namespace `ns`."""
    qname = f"{{{ns}}}{tag}"
    el = parent.find(qname)
    if el is None:
        el = etree.SubElement(parent, qname)
    return el


def fix_floating_anchor(anchor):
    """Convert wp:wrapNone + behindDoc=0 to wp:wrapTopAndBottom + behindDoc=1
    so the textbox no longer sits on top of body text."""
    # 1. Replace wp:wrapNone with wp:wrapTopAndBottom
    wrap_none = anchor.find(f"{{{WP_NS}}}wrapNone")
    if wrap_none is not None:
        wrap_topbottom = etree.Element(f"{{{WP_NS}}}wrapTopAndBottom")
        anchor.replace(wrap_none, wrap_topbottom)

    # 2. Set behindDoc="1" so the box is drawn BEHIND any incidental text
    anchor.set("behindDoc", "1")
    # And disable explicit overlap so Word respects spacing
    anchor.set("allowOverlap", "0")


def add_keep_next(p_element):
    """Add <w:keepNext/> to a paragraph's pPr so it stays with the next
    paragraph (caption)."""
    pPr = p_element.find(f"{{{W_NS}}}pPr")
    if pPr is None:
        pPr = etree.SubElement(p_element, f"{{{W_NS}}}pPr")
        # pPr must come before other children
        p_element.insert(0, pPr)
    # keepNext
    if pPr.find(f"{{{W_NS}}}keepNext") is None:
        kn = etree.SubElement(pPr, f"{{{W_NS}}}keepNext")
        kn.set(f"{{{W_NS}}}val", "1")
    # keepLines for safety (don't split the image paragraph itself)
    if pPr.find(f"{{{W_NS}}}keepLines") is None:
        kl = etree.SubElement(pPr, f"{{{W_NS}}}keepLines")
        kl.set(f"{{{W_NS}}}val", "1")


def add_spacing(p_element, before=0, after=120):
    """Add space-before / space-after (in twentieths-of-a-point) to a paragraph
    via pPr."""
    pPr = p_element.find(f"{{{W_NS}}}pPr")
    if pPr is None:
        pPr = etree.SubElement(p_element, f"{{{W_NS}}}pPr")
        p_element.insert(0, pPr)
    spacing = pPr.find(f"{{{W_NS}}}spacing")
    if spacing is None:
        spacing = etree.SubElement(pPr, f"{{{W_NS}}}spacing")
    spacing.set(f"{{{W_NS}}}before", str(before))
    spacing.set(f"{{{W_NS}}}after", str(after))


def fix():
    print("=" * 70)
    print("Fixing figure-text overlap in 論文＿李昇峰_v18.docx ...")
    print("=" * 70)

    if not SRC.exists():
        raise SystemExit(f"  ✗ Source not found: {SRC}")

    # Copy then modify
    shutil.copy(str(SRC), str(OUT))
    doc = Document(str(OUT))
    body = doc.element.body

    # ── Step 1: fix the floating cover textbox ──────────────────────────────
    anchors = body.findall(".//" + f"{{{WP_NS}}}anchor")
    print(f"  Found {len(anchors)} floating (anchored) elements")
    for anchor in anchors:
        wrap_none = anchor.find(f"{{{WP_NS}}}wrapNone")
        before_state = (
            f"behindDoc={anchor.get('behindDoc')!r}, "
            f"wrap={'wrapNone' if wrap_none is not None else 'other'}"
        )
        fix_floating_anchor(anchor)
        wrap_none_after = anchor.find(f"{{{WP_NS}}}wrapNone")
        after_state = (
            f"behindDoc={anchor.get('behindDoc')!r}, "
            f"wrap={'wrapTopAndBottom' if wrap_none_after is None else 'wrapNone'}"
        )
        print(f"    • before: {before_state}")
        print(f"      after:  {after_state}")

    # ── Step 2: figure paragraphs — keepNext + breathing room ───────────────
    n_fig_paras = 0
    n_cap_paras = 0
    paragraphs = list(body.iter(f"{{{W_NS}}}p"))
    for i, p in enumerate(paragraphs):
        has_drawing = p.find(".//" + f"{{{W_NS}}}drawing") is not None
        if not has_drawing:
            continue
        # this is a figure paragraph
        add_keep_next(p)
        add_spacing(p, before=120, after=60)  # 6pt before, 3pt after
        n_fig_paras += 1
        # next paragraph is typically the caption — also keep it together
        if i + 1 < len(paragraphs):
            cap = paragraphs[i + 1]
            cap_text = "".join(cap.itertext())
            if cap_text.startswith("圖 ") or cap_text.startswith("圖3") or cap_text.startswith("圖3-"):
                # Add keepLines and a bit of space after for breathing room
                pPr = cap.find(f"{{{W_NS}}}pPr")
                if pPr is None:
                    pPr = etree.SubElement(cap, f"{{{W_NS}}}pPr")
                    cap.insert(0, pPr)
                if pPr.find(f"{{{W_NS}}}keepLines") is None:
                    kl = etree.SubElement(pPr, f"{{{W_NS}}}keepLines")
                    kl.set(f"{{{W_NS}}}val", "1")
                add_spacing(cap, before=0, after=240)  # 12pt after caption
                n_cap_paras += 1
    print(f"  ✓ Updated {n_fig_paras} figure paragraphs (keepNext + spacing)")
    print(f"  ✓ Updated {n_cap_paras} caption paragraphs (keepLines + spacing)")

    doc.save(str(OUT))
    print(f"\n  ✓ 已儲存：{OUT}")
    print(f"  ✓ 大小：{os.path.getsize(OUT)/1024:.1f} KB")


if __name__ == "__main__":
    fix()
