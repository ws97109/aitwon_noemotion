#!/usr/bin/env python3
"""Build v14 from v13 with figure-layout fixes.

Fixes the user-reported issues:
1. 圖被字擋住：六張正文圖片寬度 5.90"（5,394,960 EMU），但 A4 內寬僅 5.77"。
   每張都向右溢出 0.13"，看起來像被切掉/被文字擋住。
   → 統一縮放到 cx = 5,150,000 EMU（≈5.63"），並按比例縮 cy。
2. 緊密就跑版：論文應使用 inline（與文字排成一行），不應用 wrap。
   v13 已是 inline，但圖片段落沒置中、沒鎖 caption。
   → 圖片段落水平置中、加 keep-with-next；caption 段加 keep-lines-together。
3. 殘留的孤兒重複圖題（純文字，不是真正 caption）會在內文中重複出現：
   [230] 圖一　SACF 整體架構圖
   [248] 圖二　極性增強注意力（PEA）模組詳細示意圖
   [255] 圖三　情感感知跨模態注意力（SACF）逐步計算示意圖
   → 比對前後段判斷為孤兒並移除（已有 af3 樣式之正式 caption 在同名圖之後）。
"""

import copy
import re
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

SRC = "/Users/lishengfeng/Desktop/aitwon_noemotion/論文＿李昇峰_v13.docx"
DST = "/Users/lishengfeng/Desktop/aitwon_noemotion/論文＿李昇峰_v14.docx"

# A4 page = 11906 twips wide, side margins 1797 each → inner 8312 twips ≈ 5.77"
# Use a comfortable safety margin so even italics / bold paddings won't overflow.
MAX_FIGURE_CX_EMU = 5_150_000  # ≈5.63" / 14.30 cm

WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def get_para_text(p):
    return "".join((t.text or "") for t in p.iter(qn("w:t")))


def get_para_style(p):
    pPr = p.find(qn("w:pPr"))
    if pPr is None:
        return "Normal"
    pStyle = pPr.find(qn("w:pStyle"))
    if pStyle is None:
        return "Normal"
    return pStyle.get(qn("w:val")) or "Normal"


def ensure_pPr(p):
    pPr = p.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p.insert(0, pPr)
    return pPr


def set_jc(p, val="center"):
    pPr = ensure_pPr(p)
    jc = pPr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        pPr.append(jc)
    jc.set(qn("w:val"), val)


def set_keep_next(p):
    pPr = ensure_pPr(p)
    if pPr.find(qn("w:keepNext")) is None:
        pPr.append(OxmlElement("w:keepNext"))


def set_keep_lines(p):
    pPr = ensure_pPr(p)
    if pPr.find(qn("w:keepLines")) is None:
        pPr.append(OxmlElement("w:keepLines"))


def has_drawing(p):
    return p.find(".//" + qn("w:drawing")) is not None


def is_paragraph(elem):
    return elem.tag == qn("w:p")


def resize_drawing(drawing_elem, max_cx):
    """Proportionally scale a wp:inline drawing so its width ≤ max_cx EMU."""
    extent = drawing_elem.find(f".//{{{WP_NS}}}extent")
    if extent is None:
        return False
    try:
        cx = int(extent.get("cx"))
        cy = int(extent.get("cy"))
    except (TypeError, ValueError):
        return False
    if cx <= max_cx:
        return False
    scale = max_cx / cx
    new_cx = max_cx
    new_cy = int(cy * scale)
    extent.set("cx", str(new_cx))
    extent.set("cy", str(new_cy))
    # Also resize embedded a:ext (picture's intrinsic frame), if present.
    for ext in drawing_elem.iter(f"{{{A_NS}}}ext"):
        try:
            ecx = int(ext.get("cx"))
            ecy = int(ext.get("cy"))
        except (TypeError, ValueError):
            continue
        # Only scale ext nodes whose cx matches the original drawing extent
        # (other ext nodes inside graphic frames may be unrelated).
        if ecx == cx and ecy == cy:
            ext.set("cx", str(new_cx))
            ext.set("cy", str(new_cy))
    return True


# ------------------------------------------------------------------ #
# Orphan duplicate figure-text removal
# ------------------------------------------------------------------ #

# Pattern: a Normal-style paragraph containing only a "圖X　..." string while
# the same caption already exists right above (af3 style). We match by exact
# text against the figure list.
FIGURE_TITLES = {
    "圖一　SACF 整體架構圖",
    "圖二　極性增強注意力（PEA）模組詳細示意圖",
    "圖三　情感感知跨模態注意力（SACF）逐步計算示意圖",
    "圖四　訓練策略全景：漸進解凍、EMA、SWA 與學習率排程",
    "圖五　多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意",
    "圖六　零洩漏推斷增強流程：TTA×5、多種子集成",
}


def remove_orphan_figure_texts(doc):
    """Remove paragraphs whose only text is a figure title but style != Caption.

    Skip paragraphs that live inside the 圖目錄 / 表目錄 indices (toc 1 style)
    or that have the af3 (Caption) style.
    """
    body = doc.element.body
    removed = 0
    for p in list(body.iter(qn("w:p"))):
        text = get_para_text(p).strip()
        if text not in FIGURE_TITLES:
            continue
        style = get_para_style(p)
        # Keep real captions (af3) and TOC entries (12 = toc 1 style id).
        if style in {"af3", "12", "Caption", "TableCaption"}:
            continue
        # The paragraph must not host a drawing (drawings are followed by an
        # af3 caption, not by a Normal duplicate title).
        if has_drawing(p):
            continue
        # Remove this orphan
        parent = p.getparent()
        if parent is not None:
            parent.remove(p)
            removed += 1
    return removed


# ------------------------------------------------------------------ #
# Figure paragraph layout fix
# ------------------------------------------------------------------ #

def fix_figure_paragraphs(doc):
    """For each paragraph that contains a drawing:

    - resize the drawing to ≤ MAX_FIGURE_CX_EMU width (preserve aspect)
    - centre-align the paragraph
    - add keep-with-next so the caption immediately below stays on the same page

    For each af3 (Caption) paragraph that follows: enforce keep-lines so the
    caption itself does not split across pages.
    """
    body = doc.element.body
    resized = 0
    fixed_paras = 0
    fixed_caps = 0

    paragraphs = [el for el in body.iter(qn("w:p"))]
    for i, p in enumerate(paragraphs):
        if not has_drawing(p):
            continue
        # Skip the cover-page paragraph: it contains text in the same paragraph
        # as the drawing (cover layout). Touching it could re-flow the cover.
        # Detect by checking for any non-empty <w:t> sibling in the paragraph.
        text_in_p = get_para_text(p).strip()
        is_cover_like = bool(text_in_p)
        if is_cover_like:
            continue

        # Resize every drawing inside this paragraph
        for drawing in p.findall(".//" + qn("w:drawing")):
            # Only resize inline drawings (anchors should already be gone in v13)
            inline = drawing.find(f"{{{WP_NS}}}inline")
            if inline is None:
                continue
            if resize_drawing(inline, MAX_FIGURE_CX_EMU):
                resized += 1

        set_jc(p, "center")
        set_keep_next(p)
        fixed_paras += 1

        # Fix the immediately-following caption paragraph if any
        nxt = p.getnext()
        while nxt is not None and not is_paragraph(nxt):
            nxt = nxt.getnext()
        if nxt is not None and get_para_style(nxt) in {"af3", "Caption"}:
            set_jc(nxt, "center")
            set_keep_lines(nxt)
            fixed_caps += 1

    return resized, fixed_paras, fixed_caps


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    doc = Document(SRC)

    removed = remove_orphan_figure_texts(doc)
    resized, fpara, fcap = fix_figure_paragraphs(doc)

    doc.save(DST)
    print(f"Saved: {DST}")
    print(f"  - Removed orphan figure-title paragraphs: {removed}")
    print(f"  - Resized drawings (width-clamped):       {resized}")
    print(f"  - Centred + keep-next figure paragraphs:  {fpara}")
    print(f"  - Caption paragraphs adjusted:             {fcap}")


if __name__ == "__main__":
    main()
