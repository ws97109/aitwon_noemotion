"""
更新論文＿李昇峰_v2.docx 圖題與圖目錄
- 將 Figure 3.2.X.  改為 圖一~圖十（Caption 樣式）
- 將 圖表 1/2 改為 圖十一/十二
- 在圖目錄新增圖一~圖十條目
- 修改 TOC 欄位指令以自動收集所有 Caption 段落
"""

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'

# 圖一~圖十：在文件本體的 Figure 段落
BODY_FIGURES = [
    (146, 'Figure 3.2.1.  ', '圖一 ', '圖 3.2.1 ', 'SACF 整體架構圖'),
    (162, 'Figure 3.2.2.  ', '圖二 ', '圖 3.2.2 ', '極性增強注意力（PEA）模組詳細示意圖'),
    (170, 'Figure 3.2.3.  ', '圖三 ', '圖 3.2.3 ', '情感感知跨模態注意力（SACF）逐步計算示意圖'),
    (177, 'Figure 3.2.4.  ', '圖四 ', '圖 3.2.4 ', '訓練策略全景：漸進解凍、EMA、SWA 與學習率排程'),
    (189, 'Figure 3.2.5.  ', '圖五 ', '圖 3.2.5 ', '多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意'),
    (196, 'Figure 3.2.6.  ', '圖六 ', '圖 3.2.6 ', '零洩漏推斷增強流程：TTA×3、多種子集成、貝葉斯先驗修正'),
    (211, 'Figure 3.2.7.  ', '圖七 ', '圖 3.2.7 ', '各版本模型性能演進對比（零洩漏條件下）'),
    (214, 'Figure 3.2.8.  ', '圖八 ', '圖 3.2.8 ', '各資料集劃分的類別先驗分布與先驗對數比率'),
    (217, 'Figure 3.2.9.  ', '圖九 ', '圖 3.2.9 ', '先驗修正縮放係數 α 搜尋：驗證集最優化與測試集效果分析'),
    (220, 'Figure 3.2.10. ', '圖十 ', '圖 3.2.10 ', '各種子結果、最終指標彙整與累積改進瀑布圖'),
]

# 圖十一/十二：已有 Caption 樣式
CAPTION_FIXES = [
    (237, '圖十一 記憶檢索關聯'),
    (248, '圖十二 人物介紹'),
]

# 圖目錄新增條目（圖一~圖十）
TOF_NEW_ENTRIES = [
    '圖一　　SACF 整體架構圖',
    '圖二　　極性增強注意力（PEA）模組詳細示意圖',
    '圖三　　情感感知跨模態注意力（SACF）逐步計算示意圖',
    '圖四　　訓練策略全景：漸進解凍、EMA、SWA 與學習率排程',
    '圖五　　多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意',
    '圖六　　零洩漏推斷增強流程：TTA×3、多種子集成、貝葉斯先驗修正',
    '圖七　　各版本模型性能演進對比（零洩漏條件下）',
    '圖八　　各資料集劃分的類別先驗分布與先驗對數比率',
    '圖九　　先驗修正縮放係數 α 搜尋：驗證集最優化與測試集效果分析',
    '圖十　　各種子結果、最終指標彙整與累積改進瀑布圖',
]


# ── 工具函式 ───────────────────────────────────────────────────

def replace_in_xml_wt(para_elem, old_str, new_str):
    """逐一檢查每個 w:t，若含有 old_str 就替換（針對單一 w:t 即含完整字串的情況）。"""
    replaced = False
    for t in para_elem.findall('.//{%s}t' % W):
        if t.text and old_str in t.text:
            t.text = t.text.replace(old_str, new_str, 1)
            replaced = True
    return replaced


def merge_and_set_wt(para_elem, new_text):
    """
    將段落內所有 w:t 的文字合併後替換成 new_text：
    第一個 w:t 設為 new_text，其餘清空。
    """
    t_elems = para_elem.findall('.//{%s}t' % W)
    if not t_elems:
        return False
    t_elems[0].text = new_text
    # 保留空格屬性
    if new_text and (new_text[0] == ' ' or new_text[-1] == ' '):
        t_elems[0].set(XML_SPACE, 'preserve')
    for t in t_elems[1:]:
        t.text = ''
    return True


def set_caption_style(para, doc):
    """將段落樣式改為 Caption（style ID = af3）。"""
    pPr = para._p.find(qn('w:pPr'))
    if pPr is None:
        pPr = OxmlElement('w:pPr')
        para._p.insert(0, pPr)
    pStyle = pPr.find(qn('w:pStyle'))
    if pStyle is None:
        pStyle = OxmlElement('w:pStyle')
        pPr.insert(0, pStyle)
    pStyle.set(qn('w:val'), 'af3')


def create_tof_para(text):
    """建立一個 table of figures 樣式的簡單段落（style ID = afd）。"""
    new_p = OxmlElement('w:p')
    # pPr with style
    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), 'afd')
    pPr.append(pStyle)
    new_p.append(pPr)
    # run + text
    new_r = OxmlElement('w:r')
    new_t = OxmlElement('w:t')
    new_t.text = text
    new_t.set(XML_SPACE, 'preserve')
    new_r.append(new_t)
    new_p.append(new_r)
    return new_p


# ── 主程式 ────────────────────────────────────────────────────

def main():
    doc = Document('論文＿李昇峰_v2.docx')
    paras = doc.paragraphs

    # ── Step 1: 更新本體圖題（Figure 3.2.X. → 圖X） ──────────────
    print('=== Step 1: 更新本體圖題段落 ===')
    for (pidx, old_prefix, new_prefix, old_inline, title) in BODY_FIGURES:
        para = paras[pidx]
        # 改 Caption 樣式
        set_caption_style(para, doc)
        # 替換 "Figure 3.2.X.  " → "圖X "
        ok1 = replace_in_xml_wt(para._p, old_prefix, new_prefix)
        if not ok1:
            # 嘗試不帶尾隨空格的版本
            ok1 = replace_in_xml_wt(para._p, old_prefix.rstrip(), new_prefix.rstrip())
        # 替換描述行內的 "圖 3.2.X " → "圖X "
        ok2 = replace_in_xml_wt(para._p, old_inline, new_prefix) if old_inline else False
        preview = para.text.replace('\n', '↵')[:80]
        print(f'  [{pidx}] {preview}  (prefix={ok1}, inline={ok2})')

    # ── Step 2: 修正 Caption 段落 237, 248（w:t 分散問題） ────────
    print()
    print('=== Step 2: 修正 Caption 段落 237 / 248 ===')
    for (pidx, new_text) in CAPTION_FIXES:
        para = paras[pidx]
        ok = merge_and_set_wt(para._p, new_text)
        print(f'  [{pidx}] {para.text[:60]}  ({"OK" if ok else "FAIL"})')

    # ── Step 3: 修正圖目錄的現有條目（56, 57）── ──────────────────
    print()
    print('=== Step 3: 修正圖目錄現有條目 ===')
    para56 = paras[56]
    para57 = paras[57]
    merge_and_set_wt(para56._p, '圖十一　　記憶檢索關聯')
    merge_and_set_wt(para57._p, '圖十二　　人物介紹')
    print(f'  [56] {para56.text[:60]}')
    print(f'  [57] {para57.text[:60]}')

    # ── Step 4: 在圖目錄的現有兩條目前插入 圖一~圖十 ────────────
    print()
    print('=== Step 4: 新增圖一~圖十至圖目錄 ===')
    # 逆序插入，使最後順序正確（圖一在最前）
    for entry_text in reversed(TOF_NEW_ENTRIES):
        new_p = create_tof_para(entry_text)
        para56._p.addprevious(new_p)
        print(f'  [+] {entry_text}')

    # ── Step 5: 修改 TOC 欄位指令，改為收集所有 Caption 段落 ─────
    print()
    print('=== Step 5: 修改 TOC 欄位指令 ===')
    old_instr = ' TOC \\h \\z \\c "圖表" '
    new_instr = ' TOC \\h \\z \\t "Caption,1" '
    for elem in para56._p.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'instrText' and elem.text == old_instr:
            elem.text = new_instr
            print(f'  TOC 指令已更新為: {new_instr}')
            break
    else:
        print('  (未找到舊指令，跳過)')

    # ── 儲存 ───────────────────────────────────────────────────
    output = '論文＿李昇峰_v3.docx'
    doc.save(output)
    print(f'\n完成！儲存至 {output}')
    print('\n提示：在 Word 中開啟後，請選取圖目錄區域 → 右鍵 → 更新欄位 → 更新整個目錄')
    print('      這樣頁碼與圖一~圖十二的超連結將會自動更新。')


if __name__ == '__main__':
    main()
