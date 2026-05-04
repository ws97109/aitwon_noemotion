"""
fix_caption_dedup.py
修正 v11 中重複的圖說標題：
- 每張圖保留一個 af3 標題（圖後位置優先，並清理文字）
- 重複的改回 Normal 樣式
- 輸出 v12
"""
import zipfile, shutil, os
from lxml import etree

SRC = "論文＿李昇峰_v11.docx"
DST = "論文＿李昇峰_v12.docx"

W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DWD = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
def qn(tag): return f"{{{W}}}{tag}"
def get_text(elem):
    return "".join(t.text for t in elem.iter(qn("t")) if t.text)

# ── 保留的 af3 標題 + 正確清理後的文字 ───────────────────────────
KEEP_CAPTIONS = {
    204: "圖一　SACF 整體架構圖",
    222: "圖二　極性增強注意力（PEA）模組詳細示意圖",
    230: "圖三　情感感知跨模態注意力（SACF）逐步計算示意圖",
    238: "圖四　訓練策略全景：漸進解凍、EMA、SWA 與學習率排程",
    252: "圖五　多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意",
    259: "圖六　零洩漏推斷增強流程：TTA×5、多種子集成",
    271: "圖七　各版本模型性能演進對比（零洩漏條件下）",  # 原誤標為圖六，更正
    278: "圖八　各種子結果、最終指標彙整與累積改進瀑布圖",
    297: "圖九　記憶檢索關聯",
    308: "圖十　人物介紹",
}

# ── 改回 Normal 的重複標題 index ────────────────────────────────
REVERT_TO_NORMAL = [216, 234, 242, 250, 264, 283, 290, 309, 320]

shutil.copy2(SRC, DST)
with zipfile.ZipFile(DST, "r") as zin:
    doc_xml = zin.read("word/document.xml")

tree = etree.fromstring(doc_xml)
body = tree.find(qn("body"))
children = list(body)

print("=== 保留並清理正確圖說標題 ===")
for idx, clean_title in KEEP_CAPTIONS.items():
    elem = children[idx]
    # 確保 af3 樣式
    pPr = elem.find(qn("pPr"))
    if pPr is None:
        pPr = etree.Element(qn("pPr"))
        elem.insert(0, pPr)
    pStyle = pPr.find(qn("pStyle"))
    if pStyle is None:
        pStyle = etree.SubElement(pPr, qn("pStyle"))
    pStyle.set(qn("val"), "af3")
    # 清理文字
    runs = elem.findall(qn("r"))
    for r in runs:
        t = r.find(qn("t"))
        if t is not None: t.text = ""
    if runs:
        first_t = runs[0].find(qn("t"))
        if first_t is None:
            first_t = etree.SubElement(runs[0], qn("t"))
        first_t.text = clean_title
        first_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    print(f"  [{idx}] ✅ {clean_title}")

print("\n=== 重複標題改回 Normal ===")
for idx in REVERT_TO_NORMAL:
    elem = children[idx]
    pPr = elem.find(qn("pPr"))
    if pPr is not None:
        pStyle = pPr.find(qn("pStyle"))
        if pStyle is not None:
            old = pStyle.get(qn("val"), "")
            if old == "af3":
                pStyle.set(qn("val"), "a")  # Normal style
                t = get_text(elem).strip()
                print(f"  [{idx}] → Normal | {t[:50]}")

print("\n=== 驗證最終 af3 段落 ===")
children = list(body)
for i, elem in enumerate(children):
    style_el = elem.find(f".//{qn('pStyle')}")
    style = style_el.get(qn("val"), "") if style_el is not None else ""
    if style == "af3":
        t = get_text(elem).strip()
        print(f"  [{i}] {t[:70]}")

# 寫回
new_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
tmp = DST + ".tmp"
with zipfile.ZipFile(DST, "r") as zin:
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item.filename))
os.replace(tmp, DST)
print(f"\n✅ 輸出：{DST}")
