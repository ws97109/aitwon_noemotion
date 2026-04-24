"""
更新論文＿李昇峰_v2.docx 的圖目錄與圖題
1. 將 Figure 3.2.X. 改為 圖一、圖二…
2. 將 Caption 式段落 圖表1/2 也更名為 圖十一、圖十二
3. 將所有圖加入圖目錄
"""

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ─── 圖片定義（按文件出現順序） ───────────────────────────────
FIGURES = [
    # (段落索引, 舊前綴, 中文名, 內文舊編號, 圖題短名)
    (146, 'Figure 3.2.1.',  '圖一',   '圖 3.2.1',  'SACF 整體架構圖'),
    (162, 'Figure 3.2.2.',  '圖二',   '圖 3.2.2',  '極性增強注意力（PEA）模組詳細示意圖'),
    (170, 'Figure 3.2.3.',  '圖三',   '圖 3.2.3',  '情感感知跨模態注意力（SACF）逐步計算示意圖'),
    (177, 'Figure 3.2.4.',  '圖四',   '圖 3.2.4',  '訓練策略全景：漸進解凍、EMA、SWA 與學習率排程'),
    (189, 'Figure 3.2.5.',  '圖五',   '圖 3.2.5',  '多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意'),
    (196, 'Figure 3.2.6.',  '圖六',   '圖 3.2.6',  '零洩漏推斷增強流程：TTA×3、多種子集成、貝葉斯先驗修正'),
    (211, 'Figure 3.2.7.',  '圖七',   '圖 3.2.7',  '各版本模型性能演進對比（零洩漏條件下）'),
    (214, 'Figure 3.2.8.',  '圖八',   '圖 3.2.8',  '各資料集劃分的類別先驗分布與先驗對數比率'),
    (217, 'Figure 3.2.9.',  '圖九',   '圖 3.2.9',  '先驗修正縮放係數 α 搜尋：驗證集最優化與測試集效果分析'),
    (220, 'Figure 3.2.10.', '圖十',   '圖 3.2.10', '各種子結果、最終指標彙整與累積改進瀑布圖'),
    # 已有 Caption 樣式的兩張圖
    (237, '圖表 1',          '圖十一', None,        '記憶檢索關聯'),
    (248, '圖表 2',          '圖十二', None,        '人物介紹'),
]

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def replace_in_runs(para, old_text, new_text):
    """在段落所有 run 的文字中替換 old_text → new_text（處理跨 run 情況）。"""
    # 先取完整文字定位
    full = para.text
    if old_text not in full:
        return

    # 逐 run 累積，找到匹配區段後替換
    runs = para.runs
    accumulated = ''
    start_run = None
    start_offset = 0

    for ri, run in enumerate(runs):
        seg = run.text
        before = accumulated
        accumulated += seg

        if old_text in accumulated:
            if start_run is None:
                # 找出匹配在哪一個 run 的哪個位置開始
                idx_in_full = before.find(old_text)
                if idx_in_full == -1:
                    idx_in_full = accumulated.find(old_text)
                start_pos = len(before)
                # 找實際起始 run
                pos = 0
                for rj, r2 in enumerate(runs):
                    if pos + len(r2.text) > idx_in_full:
                        start_run = rj
                        start_offset = idx_in_full - pos
                        break
                    pos += len(r2.text)

            # 簡化：把涉及的 run 文字合起來做替換
            # 重新掃一遍做精確替換
            break

    # 重建：將所有 run 文字串起來，做替換，再填回去
    # 只替換第一次出現（避免描述文字誤改）
    texts = [r.text for r in runs]
    joined = ''.join(texts)
    new_joined = joined.replace(old_text, new_text, 1)

    # 按原 run 長度分配新文字
    pos = 0
    for run in runs:
        old_len = len(run.text)
        run.text = new_joined[pos:pos + old_len]
        pos += old_len

    # 若有長度差異（新舊長度不同），把剩餘補到最後一個 run
    if pos < len(new_joined):
        runs[-1].text += new_joined[pos:]


def smart_replace(para, old_text, new_text):
    """更精準的替換：重組 runs 文字後替換，再分配回去（允許長度變化）。"""
    full = para.text
    if old_text not in full:
        return False

    new_full = full.replace(old_text, new_text, 1)

    # 將所有 run 文字清空，然後把新文字放進第一個 run
    if not para.runs:
        return False
    for run in para.runs:
        run.text = ''
    para.runs[0].text = new_full
    return True


def replace_text_in_xml(para, old_str, new_str):
    """直接在 XML w:t 節點做文字替換（最可靠）。"""
    replaced = False
    for t_elem in para._p.findall('.//{%s}t' % W_NS):
        if t_elem.text and old_str in t_elem.text:
            t_elem.text = t_elem.text.replace(old_str, new_str, 1)
            replaced = True
    return replaced


def set_caption_style(para, doc):
    """把段落樣式改為 Caption。"""
    try:
        para.style = doc.styles['Caption']
    except Exception:
        # fallback: 手動設 pStyle
        pPr = para._p.find(qn('w:pPr'))
        if pPr is None:
            pPr = OxmlElement('w:pPr')
            para._p.insert(0, pPr)
        pStyle = pPr.find(qn('w:pStyle'))
        if pStyle is None:
            pStyle = OxmlElement('w:pStyle')
            pPr.insert(0, pStyle)
        pStyle.set(qn('w:val'), 'Caption')


def clone_tof_para(ref_para):
    """複製一個圖目錄段落（table of figures 樣式）。"""
    new_p = copy.deepcopy(ref_para._p)
    # 清空文字
    for t in new_p.findall('.//{%s}t' % W_NS):
        t.text = ''
    return new_p


def set_tof_text(new_p, text):
    """在複製的 TOF 段落元素中設置文字。"""
    t_elems = new_p.findall('.//{%s}t' % W_NS)
    if t_elems:
        t_elems[0].text = text
        # 清掉其他 t 節點
        for t in t_elems[1:]:
            t.text = ''
    else:
        # 建一個新 run
        r = OxmlElement('w:r')
        t = OxmlElement('w:t')
        t.text = text
        r.append(t)
        new_p.append(r)


def main():
    doc = Document('論文＿李昇峰_v2.docx')
    paras = doc.paragraphs

    print('=== Step 1: 更新圖題段落 ===')
    for (pidx, old_prefix, zh_name, old_inline, short_title) in FIGURES:
        para = paras[pidx]
        current_text = para.text

        if pidx in (237, 248):
            # 已有 Caption 樣式，只改名稱
            success = replace_text_in_xml(para, old_prefix, zh_name)
            print(f'  [{pidx}] Caption 更名: "{para.text[:60]}"  ({"OK" if success else "SKIP"})')
        else:
            # 1. 改 Caption 樣式
            set_caption_style(para, doc)
            # 2. 替換 "Figure 3.2.X.  " → "圖X "
            replaced1 = replace_text_in_xml(para, old_prefix + '  ', zh_name + ' ')
            if not replaced1:
                replaced1 = replace_text_in_xml(para, old_prefix, zh_name)
            # 3. 替換描述文內的 "圖 3.2.X " → "圖X "（避免改到別的）
            if old_inline:
                replace_text_in_xml(para, old_inline + ' ', zh_name + ' ')
            print(f'  [{pidx}] 圖題更新: "{para.text[:70]}"')

    print()
    print('=== Step 2: 更新圖目錄 ===')

    # 取得兩個現有的 table of figures 段落（索引 56, 57）
    tof56 = paras[56]  # 圖表 1 記憶檢索關聯
    tof57 = paras[57]  # 圖表 2 人物介紹

    # 先更新現有兩筆（它們在目錄底部，對應 圖十一、圖十二）
    replace_text_in_xml(tof56, '圖表 1 記憶檢索關聯', '圖十一 記憶檢索關聯')
    replace_text_in_xml(tof56, '\t11', '')  # 移除舊頁碼
    replace_text_in_xml(tof57, '圖表 2 人物介紹', '圖十二 人物介紹')
    replace_text_in_xml(tof57, '\t12', '')  # 移除舊頁碼
    print(f'  [56] {tof56.text}')
    print(f'  [57] {tof57.text}')

    # 在 tof56 之前插入 圖一～圖十（共 10 筆）
    # 使用 tof56 作為樣板，逆序插入（每次都插在 tof56 前面）
    tof_entries = [
        '圖一  SACF 整體架構圖',
        '圖二  極性增強注意力（PEA）模組詳細示意圖',
        '圖三  情感感知跨模態注意力（SACF）逐步計算示意圖',
        '圖四  訓練策略全景：漸進解凍、EMA、SWA 與學習率排程',
        '圖五  多工損失函數設計：組成結構、序數懲罰矩陣與 EMD 示意',
        '圖六  零洩漏推斷增強流程：TTA×3、多種子集成、貝葉斯先驗修正',
        '圖七  各版本模型性能演進對比（零洩漏條件下）',
        '圖八  各資料集劃分的類別先驗分布與先驗對數比率',
        '圖九  先驗修正縮放係數 α 搜尋：驗證集最優化與測試集效果分析',
        '圖十  各種子結果、最終指標彙整與累積改進瀑布圖',
    ]

    # 逆序插入，使最終順序正確（圖一在最上面）
    for entry_text in reversed(tof_entries):
        new_p = clone_tof_para(tof56)
        set_tof_text(new_p, entry_text)
        tof56._p.addprevious(new_p)
        print(f'  [新增] {entry_text}')

    # 儲存
    output = '論文＿李昇峰_v3.docx'
    doc.save(output)
    print(f'\n完成！已儲存至 {output}')


if __name__ == '__main__':
    main()
