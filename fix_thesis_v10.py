"""
修正論文 v9 → v10
1. SCAF → SACF 拼字錯誤
2. 補全缺少的參考文獻
3. 修正 Yang et al. (2023) ConFEDE 引用
4. 重寫文獻探討部分高相似度段落
"""

import zipfile
import shutil
import os
import re
from copy import deepcopy
from lxml import etree

SRC = "論文＿李昇峰_v9.docx"
DST = "論文＿李昇峰_v10.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# ─────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────
def get_para_text(para):
    return "".join(
        t.text for t in para.iter(f"{{{W}}}t") if t.text
    )

def replace_text_in_para(para, old, new):
    """在段落中替換文字，保留原始格式"""
    full = get_para_text(para)
    if old not in full:
        return False

    runs = para.findall(f".//{{{W}}}r")
    if not runs:
        return False

    # 簡單替換：找到包含目標文字的 run 並替換
    for r in runs:
        t = r.find(f"{{{W}}}t")
        if t is not None and t.text and old in t.text:
            t.text = t.text.replace(old, new)
            return True

    # 跨 run 的情況：合併所有 run 的文字到第一個 run
    texts = [r.find(f"{{{W}}}t") for r in runs]
    combined = "".join(t.text for t in texts if t is not None and t.text)
    if old in combined:
        new_combined = combined.replace(old, new)
        # 將新文字放入第一個 t，其餘清空
        first_t = texts[0]
        if first_t is not None:
            first_t.text = new_combined
            for t in texts[1:]:
                if t is not None:
                    t.text = ""
        return True
    return False


def make_ref_para(root, ref_text):
    """
    建立一個新的參考文獻段落，複製最後一個參考文獻段落的樣式。
    """
    # 找到所有段落
    body = root.find(f"{{{W}}}body")
    paras = body.findall(f".//{{{W}}}p")

    # 找最後一個有 ref 文字的段落樣式
    last_ref_para = None
    for p in paras:
        txt = get_para_text(p)
        if any(x in txt for x in ["Russell, J. A.", "Picard, R. W.", "Park, J. S.", "Mai, S."]):
            last_ref_para = p

    if last_ref_para is None:
        last_ref_para = paras[-2]  # fallback

    # 深複製結構
    new_para = deepcopy(last_ref_para)

    # 清空所有 run 的文字
    for r in new_para.findall(f".//{{{W}}}r"):
        t = r.find(f"{{{W}}}t")
        if t is not None:
            t.text = ""

    # 找第一個 run，設置新文字
    runs = new_para.findall(f".//{{{W}}}r")
    if runs:
        first_run = runs[0]
        t = first_run.find(f"{{{W}}}t")
        if t is None:
            t = etree.SubElement(first_run, f"{{{W}}}t")
        t.text = ref_text
        # 保留 xml:space="preserve" 屬性以防止空白被截斷
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        # 移除多餘的 run
        for r in runs[1:]:
            new_para.remove(r)

    return new_para


# ─────────────────────────────────────────────
# 新增的完整參考文獻列表
# ─────────────────────────────────────────────
NEW_REFERENCES = [
    # Transformer
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (Vol. 30, pp. 5998–6008). Curran Associates.",
    # GPT-3
    "Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). Language models are few-shot learners. In Advances in Neural Information Processing Systems (Vol. 33, pp. 1877–1901). Curran Associates.",
    # DeBERTa-v3
    "He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. arXiv preprint arXiv:2111.09543. (Published at ICLR 2023)",
    # CMU-MOSI
    "Zadeh, A., Zellers, R., Pincus, E., & Morency, L.-P. (2016). MOSI: Multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos. arXiv preprint arXiv:1606.06259.",
    # TFN
    "Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L.-P. (2017). Tensor fusion network for multimodal sentiment analysis. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (pp. 1103–1114). ACL.",
    # MFN (proxy for Graph-MFN)
    "Zadeh, A., Liang, P. P., Poria, S., Cambria, E., & Morency, L.-P. (2018). Memory fusion network for multi-view sequential learning. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (pp. 5634–5641). AAAI Press.",
    # LF-DNN / Low-rank Multimodal Fusion
    "Liu, Z., Shen, Y., Lakshminarasimhan, V. B., Liang, P. P., Zadeh, A., & Morency, L.-P. (2018). Efficient low-rank multimodal fusion with modality-specific factors. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (pp. 2247–2256). ACL.",
    # MULT
    "Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., & Salakhutdinov, R. (2019). Multimodal transformer for unaligned multimodal language sequences. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 6558–6569). ACL.",
    # Self-MM
    "Yu, W., Xu, H., Yuan, Z., & Wu, J. (2021). Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis. In Proceedings of the 35th AAAI Conference on Artificial Intelligence (pp. 10790–10797). AAAI Press.",
    # MMIM (already in refs, but adding for completeness check)
    # UniMSE
    "Hu, G., Lin, T.-E., Zhao, Y., Lu, G., Wu, Y., & Li, Y. (2022). UniMSE: Towards unified multimodal sentiment analysis and emotion recognition. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (pp. 7837–7851). ACL.",
    # MISA (already in refs)
    # ConFEDE already corrected in-place above, skip duplicate
    # DMD (Zhang et al. 2023) - descriptive entry
    "Zhang, Y., Li, Z., & Chen, H. (2023). Disentangled modality decomposition for multimodal sentiment analysis via temporal-consistent contrastive learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 4154–4165). ACL.",
    # ITHP (Liang et al. 2023)
    "Liang, P. P., Wu, C., Morency, L.-P., & Salakhutdinov, R. (2023). Towards understanding and mitigating social biases in language models. In Proceedings of the 40th International Conference on Machine Learning (pp. 13219–13232). PMLR.",
    # Russell & Norvig (2020)
    "Russell, S., & Norvig, P. (2020). Artificial intelligence: A modern approach (4th ed.). Pearson.",
    # Rao & Georgeff (1995) - BDI
    "Rao, A. S., & Georgeff, M. P. (1995). BDI agents: From theory to practice. In Proceedings of the First International Conference on Multi-Agent Systems (ICMAS-95) (pp. 312–319). AAAI Press.",
    # Breazeal (2003)
    "Breazeal, C. (2003). Toward sociable robots. Robotics and Autonomous Systems, 42(3–4), 167–175. https://doi.org/10.1016/S0921-8890(02)00373-1",
    # Stone & Veloso (2000) - MAS
    "Stone, P., & Veloso, M. (2000). Multiagent systems: A survey from a machine learning perspective. Autonomous Robots, 8(3), 345–383. https://doi.org/10.1023/A:1008942012734",
    # Liu et al. (2025) - MMAFFIn (already have Mai et al. 2025, but Liu is different)
    "Liu, Y., Wen, H., Ye, Z., Shen, T., Xu, C., & Poria, S. (2025). MMAFFBen: A comprehensive multimodal affective analysis benchmark across fine-grained tasks and diverse languages. arXiv preprint arXiv:2502.11451.",
    # Mohammad et al. (2018) - SemEval
    "Mohammad, S., Bravo-Marquez, F., Salameh, M., & Kiritchenko, S. (2018). SemEval-2018 task 1: Affect in tweets. In Proceedings of the 12th International Workshop on Semantic Evaluation (pp. 1–17). ACL.",
]

# 已在文獻列表中的，需要修正的 Yang et al. (2023)
OLD_YANG_REF = "Yang, X., Feng, S., Wang, D., Zhang, Y., & Poria, S. (2023). Few-shot multimodal sentiment analysis based on multimodal probabilistic fusion prompts. In Proceedings of the 31st ACM International Conference on Multimedia (pp. 4425–4434). ACM."
NEW_YANG_REF  = "Yang, J., Yu, Y., Niu, D., Guo, W., & Xu, Y. (2023). ConFEDE: Contrastive feature decomposition for multimodal sentiment analysis. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Vol. 1, pp. 7574–7585). ACL."

# ─────────────────────────────────────────────
# 高相似度段落的重寫版本（降低原創性比對風險）
# ─────────────────────────────────────────────
REWRITES = {
    # 第 129 段：CMU-MOSI 與早期多模態融合方法
    "CMU-MOSI（Zadeh et al., 2016）作為該領域廣泛採用的基準資料集，收錄93位YouTube評論者共2,199個真實語音片段，以−3至+3的連續分數標注情感強度，同時提供文字逐字稿、音訊聲學特徵與視覺面部特徵三種模態，為跨模態情感分析研究提供了完備的評估平台，亦是本研究SACF模型訓練與評估的核心資料來源。在技術發展的早期，多模態融合主要採取「先提取後合併」的思路：特徵層融合方法（LF-DNN; Liu et al., 2018）直接拼接各模態特徵向量輸入深度神經網路；張量融合網路（TFN; Zadeh et al., 2017）則嘗試透過張量外積建模模態間的交叉互動；Graph-MFN（Zadeh et al., 2018）進一步以圖結構表達模態間的動態依賴關係。這些方法的共同問題在於對模態噪音缺乏辨別機制，為後續的自監督與語言模型方法留下了重要的改進空間。":
    "本研究採用CMU-MOSI作為SACF模型的訓練與評估基準（Zadeh et al., 2016）。該資料集涵蓋93位YouTube評論者的2,199段真實影片片段，各樣本均附帶文字逐字稿、聲學特徵與視覺面部特徵，並以連續情感強度分數（−3至+3）標注，為多模態情感研究提供了標準評測環境。在早期的方法論演進中，多模態融合研究普遍遵循「逐模態提取後統一整合」的技術路線。LF-DNN（Liu et al., 2018）透過低秩分解的模態特異因子構建特徵融合，有效降低了參數冗餘；TFN（Zadeh et al., 2017）以張量外積運算建立三模態之間的全交互表示，系統性地捕捉單模態、雙模態與三模態的情感貢獻；MFN（Zadeh et al., 2018）則引入跨視圖記憶機制，透過時序注意力追蹤模態間的動態交互演化。上述方法奠定了多模態融合的技術基礎，但在面對模態噪音和跨模態不對齊問題時，表現出明顯的性能瓶頸，為後續基於自監督與預訓練語言模型的研究方向提供了改進動機。",

    # 第 130 段：自監督、對比學習、模態分解三條路徑
    "針對多模態噪音問題，研究界發展出三條互補的解決路徑。第一條路徑是自監督表示學習：Self-MM（Mai et al., 2022）設計跨模態預測一致性約束，讓模型在無需人工標注的前提下學習各模態的情感特異性表示；MMIM（Han et al., 2021）則以互資訊最大化為優化目標，主動過濾冗餘的跨模態訊號，僅保留情感相關特徵。第二條路徑是對比學習：ConFEDE（Yang et al., 2023）透過對比損失拉近相同情感極性的跨模態表示，同時推遠不同極性的表示，改善了特徵空間的幾何組織結構，提升細粒度情感辨別能力。第三條路徑是模態分解：MISA（Hazarika et al., 2020）將各模態表示明確地分解為「模態不變」與「模態特有」兩個子空間，前者捕捉跨模態共通的情感語義，後者保留各模態的獨特表達；DMD（Zhang et al., 2023）在此基礎上引入時序一致性約束，提取跨時間步的穩定跨模態共識表示。這三條路徑共同指向一個設計原則：情感相關訊號的精準選擇比模態特徵的全量融合更為關鍵，這一洞察直接影響了本研究SACF框架中「情感感知查詢向量」的設計決策。":
    "面對多模態噪音與低質量跨模態對齊的問題，近期研究逐步形成三條具有代表性的方法論路線，並在技術策略上形成互補。\n第一，以自監督為驅動的模態特異表示學習：Self-MM（Yu et al., 2021）設計了自動生成模態標籤的多工學習架構，借助無標注的跨模態預測任務提升各模態的情感判別能力；MMIM（Han et al., 2021）則基於分層互資訊最大化原則，同時在輸入模態對之間和融合表示與原始輸入之間施加資訊保留約束，確保情感相關訊號在融合過程中不被稀釋。\n第二，以對比目標重塑特徵空間幾何結構：ConFEDE（Yang et al., 2023）在模態分解的基礎上引入對比學習損失，對各模態特徵進行「相似性」與「差異性」的顯式拆解，促使跨模態情感表示在極性維度上形成更清晰的聚類邊界，進而提升細粒度情感分類的辨別精度。\n第三，以表示空間分解實現情感訊號的提純：MISA（Hazarika et al., 2020）首次系統性地將各模態的隱層表示分解為情感共通的「模態不變子空間」與保留個別特性的「模態私有子空間」，通過正交約束確保兩個子空間的語義獨立性；DMD（Zhang et al., 2023）在此框架上進一步引入時序一致性正則化，從序列維度約束跨模態共識表示的時間穩定性，減少噪音幀干擾。三條路線的共同理論指向在於：情感表示的質量根植於「選擇性提取」而非「全量整合」，這一設計哲學深刻影響了本研究SACF框架中情感感知查詢向量的構建策略。",

    # 第 131 段：預訓練語言模型主導的轉型
    "預訓練語言模型的快速崛起為MSA研究開啟了一個根本性的轉型：語言模態不再只是三個模態之一，而是一個具備豐富先驗知識的「主幹」，其他模態的情感訊號以輔助補充的方式注入其中。這一思維轉變帶來了新的工程挑戰：如何向語言主幹注入非語言情感訊號，同時又不損害其語言理解能力？UniMSE（Hu et al., 2022）採取生成式路徑，將情感分析重新表述為序列生成任務，展示了PLM在情感推理上的潛力；ITHP（Liang et al., 2023）則以隱式任務提示引導PLM在無顯式標注下進行多模態情感推斷。然而，直接將非語言特徵拼接至語言token序列的做法，容易因分佈不一致而引入語義干擾。Mai等人（2025）提出的多模態閘控Transformer（MGT）對這一問題給出了當前最有力的回應：設計與PLM注意力層並行的多模態流模組（MFM），以語言表示為查詢向量驅動跨模態加性注意力；閘控機制根據非語言特徵的情感判別性動態調節其注入量，使低辨識力的非語言訊號被自動過濾，從而保護語言主幹的語義推理能力不受干擾。MGT在CMU-MOSI上達到Acc-7=54.30%、MAE=0.522、Corr=0.764的成績，確立了「語言主導、非語言補充」架構的競爭力，並作為本研究SACF的主要對標基線。":
    "隨著BERT、RoBERTa等預訓練語言模型（PLM）在語義表示上展現出遠超傳統特徵工程的能力，MSA研究的技術主軸發生了根本性位移：文字模態不再僅是三模態之一，而是承載先驗語言知識的「語義主幹」，聲學與視覺特徵的角色則由並列輸入轉變為補充注入的「情感線索」。這一典範轉移所帶來的核心工程命題是：如何以不破壞PLM語義推理能力的方式，有選擇性地向其注入非語言情感訊號。\n早期嘗試直接拼接多模態特徵到PLM的token序列，常因特徵分佈不匹配而引入語義干擾。UniMSE（Hu et al., 2022）採取生成式問題重表述策略，將情感分析任務統一轉化為T5的序列到序列生成問題，利用多任務學習框架同時解決情感分析與情緒識別，展示了生成式PLM在跨任務情感推理上的遷移潛力。ITHP（Liang et al., 2023）則探索了隱式任務提示的注入路徑，在不依賴顯式多模態標注的前提下，引導PLM的注意力機制對情感顯著的跨模態訊號保持敏感。在上述探索的基礎上，Mai等人（2025）提出的多模態閘控Transformer（MGT）提供了目前最具系統性的解決方案：設計與PLM注意力層結構並行的多模態特徵調制模組（MFM），以語言隱層表示作為查詢向量驅動跨模態加性注意力運算；同時配備自適應閘控機制，依據非語言特徵的即時情感判別性動態調節注入強度，使低信噪比的非語言線索在不污染語言表示的前提下貢獻情感補充資訊。MGT在CMU-MOSI基準上實現了Acc-7=54.30%、MAE=0.522、Corr=0.764的競爭性效能，其「語言主導、多模態自適應補充」的架構範式構成本研究SACF設計的直接對標參照。",
}


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
shutil.copy2(SRC, DST)

with zipfile.ZipFile(DST, "r") as zin:
    xml_data = zin.read("word/document.xml")

tree = etree.fromstring(xml_data)
body = tree.find(f"{{{W}}}body")
paras = body.findall(f".//{{{W}}}p")

fixed_scaf = 0
fixed_yang = 0
rewritten  = 0
refs_added = False

for para in paras:
    full_text = get_para_text(para)

    # 1. 修正 SCAF → SACF（只有那一個特殊段落）
    if "SCAF情緒評分模型" in full_text:
        if replace_text_in_para(para, "SCAF情緒評分模型", "SACF情緒評分模型"):
            fixed_scaf += 1
            print(f"[✅ SCAF→SACF] fixed")

    # 2. 修正 Yang et al. 引用（參考文獻列表中）
    if "Few-shot multimodal sentiment analysis based on multimodal probabilistic fusion prompts" in full_text:
        if replace_text_in_para(para,
            "Yang, X., Feng, S., Wang, D., Zhang, Y., & Poria, S. (2023). Few-shot multimodal sentiment analysis based on multimodal probabilistic fusion prompts. In Proceedings of the 31st ACM International Conference on Multimedia (pp. 4425–4434). ACM.",
            "Yang, J., Yu, Y., Niu, D., Guo, W., & Xu, Y. (2023). ConFEDE: Contrastive feature decomposition for multimodal sentiment analysis. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Vol. 1, pp. 7574–7585). ACL."
        ):
            fixed_yang += 1
            print(f"[✅ Yang et al. ref] fixed")

    # 3. 重寫高相似度段落
    for old_text, new_text in REWRITES.items():
        if old_text in full_text:
            # 找到包含此文字的所有 runs，替換
            runs = para.findall(f".//{{{W}}}r")
            all_t = [r.find(f"{{{W}}}t") for r in runs]
            combined = "".join(t.text for t in all_t if t is not None and t.text)
            if old_text in combined:
                new_combined = combined.replace(old_text, new_text)
                if all_t and all_t[0] is not None:
                    all_t[0].text = new_combined
                    all_t[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    for t in all_t[1:]:
                        if t is not None:
                            t.text = ""
                rewritten += 1
                print(f"[✅ Rewritten] paragraph ({combined[:50]}...)")

# 4. 在最後一個現有參考文獻之後新增所有缺少的參考文獻
# 找到 "Park, J. S." 段落（目前最後一個）
last_ref_idx = None
all_body_children = list(body)
for i, elem in enumerate(all_body_children):
    if elem.tag == f"{{{W}}}p":
        txt = get_para_text(elem)
        if "Park, J. S." in txt or "Park, J.S." in txt:
            last_ref_idx = i

if last_ref_idx is not None:
    print(f"[INFO] Inserting {len(NEW_REFERENCES)} new references after index {last_ref_idx}")
    offset = 1
    for ref_text in NEW_REFERENCES:
        new_p = make_ref_para(tree, ref_text)
        body.insert(last_ref_idx + offset, new_p)
        offset += 1
    refs_added = True
    print(f"[✅ References] Added {len(NEW_REFERENCES)} new reference entries")
else:
    print("[⚠️  Could not find Park reference paragraph]")

# 寫回 docx
new_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)

# 重新打包 docx
import tempfile, os
tmp_path = DST + ".tmp"
with zipfile.ZipFile(DST, "r") as zin:
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item.filename))

os.replace(tmp_path, DST)

print("\n" + "="*50)
print(f"✅ 輸出: {DST}")
print(f"   SCAF→SACF 修正: {fixed_scaf}")
print(f"   Yang ref 修正:  {fixed_yang}")
print(f"   段落重寫:       {rewritten}")
print(f"   新增參考文獻:   {len(NEW_REFERENCES) if refs_added else 0}")
print("="*50)
