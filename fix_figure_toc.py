"""
fix_figure_toc.py
把論文 v10 的圖目錄從靜態文字換成 Word TOC 欄位，
並清理圖說段落中混入的說明文字。
輸出：論文＿李昇峰_v11.docx
"""

import zipfile, shutil, os
from lxml import etree
from copy import deepcopy

SRC = "論文＿李昇峰_v10.docx"
DST = "論文＿李昇峰_v11.docx"

W  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

def qn(tag): return f"{{{W}}}{tag}"

def get_text(elem):
    return "".join(t.text for t in elem.iter(qn("t")) if t.text)

# ── 圖說標題清理規則（截斷到純標題）──────────────────────────
CAPTION_CLEAN = {
    226: "圖一　SACF 整體架構圖",
    244: "圖二　極性增強注意力（PEA）模組詳細示意圖",
    252: "圖三　情感感知跨模態注意力（SACF）逐步計算示意圖",
    260: "圖四　訓練策略全景：漸進解凍、EMA、SWA 與學習率排程",
    274: "圖五　多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意",
    281: "圖六　零洩漏推斷增強流程：TTA×5、多種子集成",
    293: "圖七　各版本模型性能演進對比（零洩漏條件下）",
    300: "圖八　各種子結果、最終指標彙整與累積改進瀑布圖",
    319: "圖九　記憶檢索關聯",
    330: "圖十　人物介紹",
}

# ── TOC 欄位 XML（Word 會在開啟時更新）─────────────────────────
TOC_FIELD_XML = """<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:pPr><w:pStyle w:val="afd"/></w:pPr>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:fldChar w:fldCharType="begin" w:dirty="true"/>
  </w:r>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:instrText xml:space="preserve"> TOC \\h \\z \\t "caption,1" </w:instrText>
  </w:r>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:fldChar w:fldCharType="separate"/>
  </w:r>
  <w:r>
    <w:t xml:space="preserve">（請在 Word 中右鍵點擊此處並選擇「更新欄位」以顯示圖目錄）</w:t>
  </w:r>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:fldChar w:fldCharType="end"/>
  </w:r>
</w:p>"""

# ── 主流程 ──────────────────────────────────────────────────────
shutil.copy2(SRC, DST)

with zipfile.ZipFile(DST, "r") as zin:
    xml_bytes = zin.read("word/document.xml")

tree = etree.fromstring(xml_bytes)
body = tree.find(qn("body"))
children = list(body)   # body 的直接子元素列表

# 1. 清理混入說明文字的圖說段落
print("=== Step 1：清理混入說明文字的圖說段落 ===")
for idx, clean_title in CAPTION_CLEAN.items():
    elem = children[idx]
    if elem.tag != qn("p"):
        print(f"  [WARN] body[{idx}] 不是段落，跳過")
        continue

    old_text = get_text(elem)
    if clean_title in old_text and old_text == clean_title:
        print(f"  [OK ] body[{idx}] 已是正確標題，無需修改")
        continue

    # 清空所有 run 的文字
    runs = elem.findall(qn("r"))
    for r in runs:
        t_el = r.find(qn("t"))
        if t_el is not None:
            t_el.text = ""

    # 設定第一個 run 的文字為乾淨標題
    if runs:
        first_t = runs[0].find(qn("t"))
        if first_t is None:
            first_t = etree.SubElement(runs[0], qn("t"))
        first_t.text = clean_title
        first_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    print(f"  [FIX] body[{idx}] → {clean_title[:60]}")

# 2. 找到靜態圖目錄位置（body index 113-122，style=afd）
print("\n=== Step 2：替換靜態圖目錄 → Word TOC 欄位 ===")
static_tof_indices = []
for i, elem in enumerate(children):
    if elem.tag == qn("p"):
        style_el = elem.find(f".//{qn('pStyle')}")
        style = style_el.get(qn("val"), "") if style_el is not None else ""
        text = get_text(elem)
        if style == "afd" and any(f"圖{c}" in text for c in "一二三四五六七八九十"):
            static_tof_indices.append(i)

print(f"  找到靜態圖目錄段落：{static_tof_indices}")

if static_tof_indices:
    # 確保是連續的
    first_idx = static_tof_indices[0]

    # 移除靜態圖目錄段落（從後往前移除避免 index 偏移）
    for i in sorted(static_tof_indices, reverse=True):
        body.remove(children[i])

    # 在原位置插入 TOC 欄位
    toc_elem = etree.fromstring(TOC_FIELD_XML)
    # 重新計算 children（移除後 index 有變化）
    children_new = list(body)
    body.insert(first_idx, toc_elem)
    print(f"  TOC 欄位插入於 body index {first_idx}")
else:
    print("  [WARN] 找不到靜態圖目錄段落")

# 3. 寫回 docx
print("\n=== Step 3：寫回 docx ===")
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

print(f"\n✅ 完成：{DST}")
print("   → 在 Word 中開啟後，對圖目錄欄位按右鍵 → 「更新欄位」即可自動生成圖目錄")
