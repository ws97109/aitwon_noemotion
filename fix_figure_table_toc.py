"""
fix_figure_table_toc.py
從 v10 建立正確的圖目錄＋表目錄（使用 Word TOC 欄位）
- 圖說段落套用 Caption 樣式 (af3)
- 表標題段落套用新建的 TableCaption 樣式 (af3t)
- 以兩個獨立的 TOC 欄位分別生成圖目錄與表目錄
輸出：論文＿李昇峰_v11.docx
"""

import zipfile, shutil, os
from lxml import etree
from copy import deepcopy

SRC = "論文＿李昇峰_v10.docx"
DST = "論文＿李昇峰_v11.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def qn(tag): return f"{{{W}}}{tag}"
def get_text(elem):
    return "".join(t.text for t in elem.iter(qn("t")) if t.text)

# ── 圖說標題（精簡版，去掉說明文字）────────────────────────────
FIGURE_CAPTION_FIXES = {
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

# ── 表標題位置（body 直接子項 index）────────────────────────────
TABLE_CAPTION_INDICES = {
    384: "表一　CMU-MOSI 多模態情感分析效能比較",
    404: "表二　SACF-Text 在 MMAFFBen 五個純文字任務上的 ma-F1 比較",
    410: "表三　SACF-Text 在 SemEval-2018 Task 1 三語言整體分數比較",
    416: "表四　MMAFFIn 領域自適應預訓練消融：v60_baseline vs v60_MMAFFIn",
    423: "表五　參數效率比較（效率分數 = 整體得分 ÷ 參數量（B））",
}

# ── TOC 欄位 XML 產生函式 ──────────────────────────────────────
def make_toc_field(style_name):
    """生成 Word TOC 欄位，引用指定 style 的段落"""
    instr = f' TOC \\h \\z \\t "{style_name},1" '
    hint  = f'（請在 Word 中對此欄位按右鍵 → 「更新欄位」以顯示目錄）'
    xml = f"""<w:p xmlns:w="{W}">
  <w:pPr><w:pStyle w:val="afd"/></w:pPr>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:fldChar w:fldCharType="begin" w:dirty="true"/>
  </w:r>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:instrText xml:space="preserve">{instr}</w:instrText>
  </w:r>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:fldChar w:fldCharType="separate"/>
  </w:r>
  <w:r><w:t xml:space="preserve">{hint}</w:t></w:r>
  <w:r><w:rPr><w:noProof/></w:rPr>
    <w:fldChar w:fldCharType="end"/>
  </w:r>
</w:p>"""
    return etree.fromstring(xml)

def make_heading(text, style="a3"):
    """生成小標題段落（圖目錄／表目錄）"""
    xml = f"""<w:p xmlns:w="{W}">
  <w:pPr><w:pStyle w:val="{style}"/></w:pPr>
  <w:r><w:t xml:space="preserve">{text}</w:t></w:r>
</w:p>"""
    return etree.fromstring(xml)

def make_blank():
    return etree.fromstring(f'<w:p xmlns:w="{W}"/>')

# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
shutil.copy2(SRC, DST)

with zipfile.ZipFile(DST, "r") as zin:
    doc_xml   = zin.read("word/document.xml")
    style_xml = zin.read("word/styles.xml")

# ─── 1. 在 styles.xml 新增 TableCaption 樣式 ─────────────────────
print("=== Step 1：新增 TableCaption 樣式 ===")
style_tree = etree.fromstring(style_xml)
# 找到現有 Caption (af3) 樣式，複製一份改成 TableCaption
caption_style = style_tree.find(
    f".//{qn('style')}[@{qn('styleId')}='af3']"
)
# 若沒找到，用 styleId="af3" 的形式再試
if caption_style is None:
    for s in style_tree.iter(qn("style")):
        sid = s.get(qn("styleId"), "")
        if sid == "af3":
            caption_style = s
            break

if caption_style is not None:
    new_style = deepcopy(caption_style)
    new_style.set(qn("styleId"), "af3t")
    # 修改 name
    name_el = new_style.find(qn("name"))
    if name_el is not None:
        name_el.set(qn("val"), "TableCaption")
    # 修改 basedOn（確保繼承正確）
    base_el = new_style.find(qn("basedOn"))
    if base_el is not None:
        base_el.set(qn("val"), "af3")
    style_tree.append(new_style)
    print("  [OK] TableCaption (af3t) 樣式已新增")
else:
    print("  [WARN] 找不到 Caption (af3) 樣式，TableCaption 建立失敗")

new_style_xml = etree.tostring(
    style_tree, xml_declaration=True, encoding="UTF-8", standalone=True
)

# ─── 2. 修改 document.xml ─────────────────────────────────────────
doc_tree = etree.fromstring(doc_xml)
body = doc_tree.find(qn("body"))
children = list(body)

# 2a. 清理圖說段落（截斷混入的說明文字），確保 af3 樣式
print("\n=== Step 2a：清理圖說段落 ===")
for idx, clean_title in FIGURE_CAPTION_FIXES.items():
    elem = children[idx]
    if elem.tag != qn("p"):
        print(f"  [SKIP] body[{idx}] 不是段落"); continue
    # 確保樣式是 af3
    pPr = elem.find(qn("pPr"))
    if pPr is None:
        pPr = etree.SubElement(elem, qn("pPr"))
        elem.insert(0, pPr)
    pStyle = pPr.find(qn("pStyle"))
    if pStyle is None:
        pStyle = etree.SubElement(pPr, qn("pStyle"))
    pStyle.set(qn("val"), "af3")
    # 設定乾淨標題文字
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
    print(f"  [FIX] body[{idx}] → {clean_title}")

# 2b. 套用 TableCaption (af3t) 到表標題段落
print("\n=== Step 2b：套用 TableCaption 樣式到表標題 ===")
for idx, clean_title in TABLE_CAPTION_INDICES.items():
    elem = children[idx]
    if elem.tag != qn("p"):
        print(f"  [SKIP] body[{idx}] 不是段落"); continue
    # 取得或建立 pPr
    pPr = elem.find(qn("pPr"))
    if pPr is None:
        pPr = etree.Element(qn("pPr"))
        elem.insert(0, pPr)
    pStyle = pPr.find(qn("pStyle"))
    if pStyle is None:
        pStyle = etree.SubElement(pPr, qn("pStyle"))
    pStyle.set(qn("val"), "af3t")
    # 標準化標題文字（去掉括號說明）
    runs = elem.findall(qn("r"))
    if runs:
        for r in runs:
            t = r.find(qn("t"))
            if t is not None: t.text = ""
        first_t = runs[0].find(qn("t"))
        if first_t is None:
            first_t = etree.SubElement(runs[0], qn("t"))
        first_t.text = clean_title
        first_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    print(f"  [FIX] body[{idx}] → {clean_title}")

# 2c. 替換靜態圖目錄＋表目錄 → 分開的兩個動態 TOC 欄位
print("\n=== Step 2c：替換靜態圖表目錄 → 動態 TOC 欄位 ===")
# 找到圖表目錄標題（style=a3，包含「圖表目錄」）
toc_heading_idx = None
for i, elem in enumerate(children):
    t = get_text(elem)
    style_el = elem.find(f".//{qn('pStyle')}")
    style = style_el.get(qn("val"), "") if style_el is not None else ""
    if "圖表目錄" in t and style == "a3":
        toc_heading_idx = i
        break

if toc_heading_idx is None:
    print("  [WARN] 找不到圖表目錄標題段落");
else:
    print(f"  [OK] 圖表目錄標題在 body[{toc_heading_idx}]")
    # 找出緊隨其後的靜態圖表目錄段落（afd 樣式）
    static_indices = []
    j = toc_heading_idx + 1
    while j < len(children):
        elem = children[j]
        style_el = elem.find(f".//{qn('pStyle')}")
        style = style_el.get(qn("val"), "") if style_el is not None else ""
        t = get_text(elem).strip()
        if style == "afd" and t:
            static_indices.append(j)
            j += 1
        elif not t:  # 空白段落也跳過
            j += 1
        else:
            break

    print(f"  [OK] 靜態目錄段落：{static_indices}")

    # 移除靜態段落（從後往前避免 index 偏移）
    for i in sorted(static_indices, reverse=True):
        body.remove(children[i])

    # 重新取 children（已移除）
    children = list(body)
    # 重新找標題位置
    for i, elem in enumerate(children):
        t = get_text(elem)
        style_el = elem.find(f".//{qn('pStyle')}")
        style = style_el.get(qn("val"), "") if style_el is not None else ""
        if "圖表目錄" in t and style == "a3":
            toc_heading_idx = i
            break

    # 在標題後插入：圖目錄標題 + 圖 TOC + 空行 + 表目錄標題 + 表 TOC
    insert_pos = toc_heading_idx + 1
    inserts = [
        make_heading("圖目錄", style="a3"),
        make_toc_field("caption"),
        make_blank(),
        make_heading("表目錄", style="a3"),
        make_toc_field("TableCaption"),
        make_blank(),
    ]
    for offset, elem in enumerate(inserts):
        body.insert(insert_pos + offset, elem)

    # 移除原 "圖表目錄" 標題（已被 圖目錄+表目錄 取代）
    children = list(body)
    for i, elem in enumerate(children):
        t = get_text(elem)
        style_el = elem.find(f".//{qn('pStyle')}")
        style = style_el.get(qn("val"), "") if style_el is not None else ""
        if t == "圖表目錄" and style == "a3":
            body.remove(elem)
            print(f"  [OK] 移除舊「圖表目錄」標題，插入「圖目錄」+「表目錄」各含 TOC 欄位")
            break

# ─── 3. 寫回 docx ────────────────────────────────────────────────
print("\n=== Step 3：寫回 docx ===")
new_doc_xml = etree.tostring(
    doc_tree, xml_declaration=True, encoding="UTF-8", standalone=True
)

tmp = DST + ".tmp"
with zipfile.ZipFile(DST, "r") as zin:
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, new_doc_xml)
            elif item.filename == "word/styles.xml":
                zout.writestr(item, new_style_xml)
            else:
                zout.writestr(item, zin.read(item.filename))
os.replace(tmp, DST)

print(f"""
{'='*55}
✅  輸出：{DST}

   • 圖一～圖十  → Caption (af3) 樣式  ✓
   • 表一～表五  → TableCaption (af3t) 樣式  ✓
   • 舊靜態清單  → 刪除  ✓
   • 新插入「圖目錄」TOC 欄位（引用 caption 樣式）  ✓
   • 新插入「表目錄」TOC 欄位（引用 TableCaption 樣式）  ✓

📌  在 Word 開啟後：
   1. Ctrl + A（全選）→ F9（更新所有欄位）
   2. 選「更新整個目錄」→ 圖目錄與表目錄自動生成
{'='*55}""")
