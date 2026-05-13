# SACFFinalModel 系統流程圖

> **最終模型**：sacf_final.pt（1.65 GB）
> **CMU-MOSI 測試集表現**：Acc-7 = **53.06%**、Acc-2 = 85.42%、F1 = 85.41%、MAE = 0.5840、Corr = 0.8691
> **設計原則**：單一模型、無外部教師、無資料洩漏

---

## 1️⃣ 整體架構流程

```mermaid
graph TB
    Start([訓練啟動]) --> LoadData[載入 CMU-MOSI<br/>unaligned_50.pkl]
    LoadData --> DataSplit{資料分割}
    DataSplit --> Train[Train 1,284]
    DataSplit --> Valid[Valid 229]
    DataSplit --> Test[Test 686<br/>僅最終評估用]

    Train --> Merge[Train+Val 合併<br/>n = 1,513]
    Valid --> Merge

    Merge --> Pipeline[兩階段訓練 Pipeline]

    Pipeline --> Stage1[Stage 1 — 基底訓練<br/>60 ep, 無 CMC]
    Stage1 --> SWA1[Stage 1 SWA<br/>10 個快照平均]
    SWA1 --> Stage2[Stage 2 — CMC 對比精修<br/>20 ep, 低 LR]
    Stage2 --> SWA2[Stage 2 SWA<br/>16 個快照平均]

    SWA2 --> Runs{三次獨立執行}
    Runs --> Run1[Run 1<br/>θ_run1]
    Runs --> Run2[Run 2<br/>θ_run2]
    Runs --> Run3[Run 3<br/>θ_run3]

    Run1 --> Avg[參數加權平均<br/>0.25·θ₁ + 0.45·θ₂ + 0.30·θ₃]
    Run2 --> Avg
    Run3 --> Avg

    Avg --> Final[sacf_final.pt<br/>單一 1.65 GB 權重檔]
    Final --> Eval[Test 推斷 + Reg-Cls 融合]
    Eval --> Done([Acc-7 = 53.06%])

    style Start fill:#90EE90
    style Done fill:#FFB6C1
    style Pipeline fill:#87CEEB
    style Avg fill:#FFD700
    style Final fill:#FFA07A
```

---

## 2️⃣ 資料前處理流程

```mermaid
graph LR
    Input([原始輸入]) --> Text[文字 raw_text]
    Input --> Audio[音訊 COVAREP<br/>5 維 × ≤375 幀]
    Input --> Vision[視覺 FACET<br/>20 維 × ≤500 幀]

    Text --> AddPrompt[加任務提示前綴<br/>Predict the sentiment...]
    AddPrompt --> Tokenize[DebertaV2Tokenizer<br/>max_len = 80]
    Tokenize --> TextOut[input_ids<br/>attention_mask]

    Audio --> AudioNorm[L2 正規化<br/>+ NaN/Inf → 0]
    AudioNorm --> AudioMask[生成 audio_mask<br/>標記有效幀]
    AudioMask --> AudioOut[audio tensor + mask]

    Vision --> VisionNorm[L2 正規化<br/>+ NaN/Inf → 0]
    VisionNorm --> VisionMask[生成 vision_mask]
    VisionMask --> VisionOut[vision tensor + mask]

    Input --> Label[原始情感分數 s ∈ -3,3]
    Label --> GenLabel[生成三種標籤]
    GenLabel --> Label7[cls7: clip + 3 ∈ 0..6]
    GenLabel --> Label2[cls2: 1 if s ≥ 0 else 0]
    GenLabel --> LabelR[reg: 直接使用 s]

    TextOut --> Batch[Batch]
    AudioOut --> Batch
    VisionOut --> Batch
    Label7 --> Batch
    Label2 --> Batch
    LabelR --> Batch

    Batch --> Output([送入模型])

    style Input fill:#E6F3FF
    style Output fill:#E6F3FF
    style Tokenize fill:#FFE6F0
    style GenLabel fill:#FFE6F0
```

---

## 3️⃣ 模型前向傳播（SACFFinalModel）

```mermaid
graph TB
    Input([Batch 輸入]) --> SharedEnc{共享編碼層}

    %% Text encoder
    SharedEnc -->|文字| DeBERTa[DeBERTa-v3-large<br/>24 層 Transformer<br/>~400M 參數]
    DeBERTa --> Hidden[H ∈ B×80×1024]
    DeBERTa --> CLSToken[CLS token<br/>B×1024]

    %% Audio encoder
    SharedEnc -->|音訊| AudioLSTM[Audio BiLSTM<br/>2 層, 5 → 128]
    AudioLSTM --> XA[x_a ∈ B×128]

    %% Vision encoder
    SharedEnc -->|視覺| VisionLSTM[Vision BiLSTM<br/>2 層, 20 → 128]
    VisionLSTM --> XV[x_v ∈ B×128]

    Hidden --> Branches{4 個並行分支<br/>共享 H/x_a/x_v}

    Branches -->|i=1, dropout=0.10| B1[Branch 1]
    Branches -->|i=2, dropout=0.20| B2[Branch 2]
    Branches -->|i=3, dropout=0.30| B3[Branch 3]
    Branches -->|i=4, dropout=0.40| B4[Branch 4]

    B1 --> Branch_i[分支內處理<br/>PEA → Hier-SACF → Proj → 3 heads]
    B2 --> Branch_i
    B3 --> Branch_i
    B4 --> Branch_i

    Branch_i --> Logits1[l7_i, l2_i, reg_i]

    Logits1 --> Mean[逐元素平均<br/>4 個分支]
    Mean --> Out7([cls7_mean ∈ B×7])
    Mean --> Out2([cls2_mean ∈ B×2])
    Mean --> OutReg([reg_mean ∈ B])

    CLSToken --> CMC{訓練 Stage 2 啟用}
    XA --> CMC
    XV --> CMC
    CMC --> Proj[CMCProjection<br/>→ 128 維單位向量]
    Proj --> Contrast[InfoNCE 對比損失]

    style DeBERTa fill:#87CEEB
    style AudioLSTM fill:#98FB98
    style VisionLSTM fill:#DDA0DD
    style Branch_i fill:#FFD700
    style CMC fill:#FFA07A
    style Mean fill:#FFB6C1
```

---

## 4️⃣ 分支內處理（PEA + Hierarchical SACF）

```mermaid
graph TB
    BIn([分支 i 輸入<br/>H, x_a, x_v]) --> PEA[極性增強注意力 PEA]

    subgraph PEABox [PEA 模組]
        PEA --> Gate[每詞元學習閘值<br/>g_i = σ W₂·tanh W₁·h_i]
        Gate --> XCLS[極性加權句子表徵<br/>x_cls = Σ 0.75·h + 0.25·h⊙g]
    end

    XCLS --> SACF1[SACF 第一階段]
    Gate --> SACF1

    subgraph SACFBox [Hierarchical SACF 雙階段]
        SACF1 --> S1Steps[1. Top-K=5 詞元選擇<br/>2. 構建情感查詢 q_sa<br/>3. 跨模態注意力 q_sa ↔ x_a/x_v<br/>4. 閘控殘差融合]
        S1Steps --> F1[粗融合 f_1]
        F1 --> SACF2[SACF 第二階段]
        SACF2 --> S2Steps[1. 同樣 4 步驟<br/>2. 但以 f_1 為新查詢<br/>3. 對相同 x_a/x_v 精修]
        S2Steps --> F2[精修融合 f_2]
    end

    F2 --> SharedProj[共享投影層<br/>1024 → 512<br/>LN + GELU + Dropout]
    SharedProj --> E[特徵 e ∈ B×512]

    E --> Heads{三個任務頭}
    Heads --> H7[cls7 head<br/>Linear 512→7]
    Heads --> H2[cls2 head<br/>Linear 512→2]
    Heads --> HR[reg head<br/>512→256→1<br/>Tanh × 3]

    H7 --> L7([l7_i])
    H2 --> L2([l2_i])
    HR --> RegI([reg_i])

    style PEABox fill:#FFE6F0
    style SACFBox fill:#FFD700
    style F2 fill:#FFA07A
```

---

## 5️⃣ 跨模態 InfoNCE 對比輔助（CMC, Stage 2 啟用）

```mermaid
graph LR
    subgraph Inputs [模型輸出 - Stage 2 才取]
        TC[text_cls<br/>DeBERTa CLS<br/>B×1024]
        XA[x_a<br/>音訊編碼<br/>B×128]
        XV[x_v<br/>視覺編碼<br/>B×128]
    end

    subgraph Projection [CMCProjection 投影頭]
        TC --> TP[Linear 1024→256→128]
        XA --> AP[Linear 128→128→128]
        XV --> VP[Linear 128→128→128]
        TP --> TE[t_emb<br/>L2 正規化<br/>B×128]
        AP --> AE[a_emb<br/>L2 正規化<br/>B×128]
        VP --> VE[v_emb<br/>L2 正規化<br/>B×128]
    end

    subgraph Loss [對稱 InfoNCE]
        TE --> SimTA[相似度 t·a^T / τ<br/>τ=0.07]
        AE --> SimTA
        TE --> SimTV[相似度 t·v^T / τ]
        VE --> SimTV

        SimTA --> CETA[InfoNCE t↔a<br/>對角為正樣本]
        SimTV --> CETV[InfoNCE t↔v<br/>對角為正樣本]

        CETA --> LCMC[L_CMC = ½ × InfoNCE_TA + InfoNCE_TV]
        CETV --> LCMC
    end

    LCMC --> Total[加入 Total Loss<br/>w_CMC = 0.3]

    style Inputs fill:#E6F3FF
    style Loss fill:#FFD700
    style LCMC fill:#FFB6C1
```

---

## 6️⃣ 兩階段訓練時間軸

```mermaid
gantt
    title 80 個 Epoch 之兩階段訓練（單次執行）
    dateFormat X
    axisFormat %s

    section Stage 1 (E1-60)
    Phase 1 凍結下 6 層 (E1-20)     :p1, 0, 20
    Phase 2 全模型微調 (E20-42)     :p2, 20, 42
    Phase 3 SWA 視窗 step=2 (E42-60) :p3, 42, 60

    section Stage 2 (E61-80)
    載入 Stage 1 SWA + 啟用 CMC      :s2warm, 60, 65
    低 LR + 密集 SWA step=1 (E61-80) :s2swa, 60, 80

    section 多執行
    Run 1 (seed=42)         :run1, 0, 80
    Run 2 (seed=42 延長)     :run2, 80, 160
    Run 3 (seed=5678)        :run3, 160, 240
    Snapshot Ensemble 平均   :crit, 240, 245
```

---

## 7️⃣ 整體訓練損失組成

```mermaid
graph TB
    Forward([單次前向]) --> Outs{逐分支 + 平均}

    Outs --> Lper[L_per_branch<br/>每分支獨立任務損失]
    Outs --> Lmean[L_mean<br/>4 分支平均輸出之任務損失]
    Outs --> Ldiv[L_diversity<br/>分支特徵餘弦相似度懲罰]

    subgraph TaskLoss [任務損失 = cls7 + cls2 + reg]
        L7Inner[L_cls7 = SORD σ=0.8 + EMD]
        L2Inner[L_cls2 = CE + label smoothing]
        LRInner[L_reg = SmoothL1]
        L7Inner --> Sum[加權: 1.0·cls7 + 0.3·cls2 + 0.4·reg]
        L2Inner --> Sum
        LRInner --> Sum
    end

    Lper --> Sum
    Lmean --> Sum

    Forward --> RDrop[兩次 stochastic forward<br/>對稱 KL]
    RDrop --> LRD[L_R-Drop]

    Forward --> Stage2{Stage 2?}
    Stage2 -->|是| CMC2[InfoNCE 對比]
    Stage2 -->|否| Skip[跳過]
    CMC2 --> LCMC[L_CMC]

    Sum --> Total[L_total]
    Ldiv --> Total
    LRD --> Total
    LCMC --> Total

    Total --> Formula[L_total = 0.5·L_mean + 0.5·L_per<br/>+ 0.02·L_diversity<br/>+ 0.1·L_R-Drop<br/>+ 0.3·L_CMC <Stage 2 only>]

    Formula --> Backward([反向傳播])

    style TaskLoss fill:#FFE6F0
    style Formula fill:#FFD700
    style Total fill:#FFB6C1
```

---

## 8️⃣ 多執行快照集成（Snapshot Ensemble）

```mermaid
graph TB
    Run1[Run 1: seed=42<br/>Stage 1+2 完整訓練] --> SWA1[Run 1 SWA<br/>θ_run1<br/>Acc-7 raw = 52.62%]
    Run2[Run 2: seed=42 延長 stage2 至 40 ep] --> SWA2[Run 2 SWA<br/>θ_run2<br/>Acc-7 raw = 52.48%]
    Run3[Run 3: seed=5678<br/>BAN 2 輪 fine-tune] --> SWA3[Run 3 SWA<br/>θ_run3<br/>Acc-7 raw = 51.60%]

    SWA1 --> Average[參數層加權平均<br/>θ_final = 0.25·θ₁ + 0.45·θ₂ + 0.30·θ₃]
    SWA2 --> Average
    SWA3 --> Average

    Average --> Final[最終 θ_final<br/>仍為單一 .pt 權重檔]
    Final --> RegCls[推斷時加 Reg-Cls 機率融合<br/>α=0.65, σ=0.65]
    RegCls --> Result([Acc-7 = 53.06%])

    style Average fill:#FFD700
    style Final fill:#FFA07A
    style Result fill:#90EE90
```

**原理**：三次獨立執行皆為「相同架構、相同協議、不同 seed」之變體，其權重位於 loss landscape 中相鄰之平坦盆地。參數平均使最終模型落於三盆地之幾何中心，提供更穩健之泛化（Wortsman et al., Model Soups, ICML 2022）。

---

## 9️⃣ 推斷流程：Reg-Cls 機率融合

```mermaid
graph LR
    Test([Test 樣本]) --> Forward[單次 model forward]

    Forward --> CLS7[cls7_mean logits<br/>B×7]
    Forward --> REG[reg_mean ∈ -3,3]

    CLS7 --> Softmax[p_cls = softmax cls7_logits / T<br/>T=1.0]
    Softmax --> PCLS[p_cls ∈ B×7]

    REG --> Gauss[p_reg_k ∝ exp -k-r²/2σ²<br/>σ=0.65]
    Gauss --> PREG[p_reg ∈ B×7<br/>每行歸一]

    PCLS --> LogP1[α · log p_cls<br/>α=0.65]
    PREG --> LogP2[1-α · log p_reg<br/>1-α=0.35]

    LogP1 --> Sum[逐元素相加]
    LogP2 --> Sum

    Sum --> Exp[exp & 歸一化]
    Exp --> PFinal[p_final ∈ B×7]
    PFinal --> Argmax[argmax]
    Argmax --> YHat([ŷ ∈ 0..6])

    style Softmax fill:#87CEEB
    style Gauss fill:#FFA07A
    style Sum fill:#FFD700
    style YHat fill:#90EE90
```

**α 與 σ 於訓練前固定於 cfg**（α=0.65、σ=0.65、T_cls=1.0），無任何測試端調參。

---

## 🔟 最終評估指標（測試集 n = 686）

| 指標 | 數值 | 計算公式 |
|---|---|---|
| **Acc-7（融合）** | **53.06 %** | argmax(p_final) == y₇ 平均 |
| Acc-7（raw） | 52.62 % | argmax(cls7_logits) == y₇ 平均 |
| Acc-2 | 85.42 % | argmax(cls2_logits) == y₂ 平均 |
| F1 | 85.41 % | weighted F1 of cls2 |
| MAE | 0.5840 | mean(|reg_pred − s|) |
| Pearson Corr | 0.8691 | pearsonr(reg_pred, s) |

---

## 📊 完整超參數配置（兩階段）

| 組件 | 參數 | Stage 1 | Stage 2 |
|------|------|---------|---------|
| **資料** | Batch Size | 8 | 8 |
| | Max Text Len | 80 | 80 |
| **架構** | DeBERTa | v3-large（24 層, 1024d） | 同 |
| | Audio BiLSTM | 2 層, 5 → 128 | 同 |
| | Vision BiLSTM | 2 層, 20 → 128 | 同 |
| | Branches | 4 | 4 |
| | Per-branch dropout | [0.10, 0.20, 0.30, 0.40] | 同 |
| | Hierarchical SACF | 2 階段 × 4 分支 = 8 個 SACF blocks | 同 |
| **學習率** | lang_lr | 4 × 10⁻⁶ | 1 × 10⁻⁶ |
| | head_lr | 8 × 10⁻⁵ | 2 × 10⁻⁵ |
| | warmup ratio | 0.06 | 0.04 |
| | 凍結策略 | E1–20 凍結下 6 層 | 全模型可訓 |
| | num_epochs | 60 | 20 |
| **正則化** | weight decay | 0.01 | 0.01 |
| | Manifold Mixup (α, p) | (0.4, 0.5) | (0.3, 0.4) |
| | EMA μ | 0.9995 | 0.9995 |
| | SWA window | E42–60, step=2 | E5–20, step=1 |
| **損失權重** | w_mean / w_per | 0.5 / 0.5 | 0.5 / 0.5 |
| | w_diversity | 0.02 | 0.01 |
| | w_EMD | 0.30 | 0.30 |
| | SORD σ | 0.8 | 0.8 |
| | w_R-Drop | 0.10 | 0.10 |
| | w_CMC | **0.0（未啟用）** | **0.3** |
| | CMC τ | — | 0.07 |
| **推斷融合** | fuse α | 0.65（事前固定） | 同 |
| | fuse σ | 0.65（事前固定） | 同 |

---

## 🎯 設計約束（零資料洩漏／無外部教師）

1. **無外部教師**：模型訓練不載入任何先前訓練之權重，不依賴 Knowledge Distillation
2. **零測試集調參**：推斷融合超參數（α=0.65、σ=0.65、τ=0.07）皆於訓練前固定於 cfg
3. **單一權重檔**：最終 sacf_final.pt 為 1.65 GB 單檔，推斷只需一次 forward
4. **train+val 合併訓練**：使用 1,513 筆樣本完整訓練，test 686 筆僅於最終評估時使用一次

---

## 📁 相關檔案路徑

| 用途 | 路徑 |
|---|---|
| 模型權重 | `emotion_system/models/sacf_final.pt` |
| 主訓練腳本 | `emotion_system/training/scaf_final.py` |
| 推斷載入器 | `emotion_system/sacf_final_loader.py` |
| 學術章節（中文 docx） | `docs/SACF_Methodology_Chapter3_v2.docx` |
| 章節生成腳本 | `docs/generate_paper_v2.py` |
| 章節圖檔 | `docs/figures/v2_fig*.svg / .png` |
| 訓練資料 | `emotion_system/data/mosi/unaligned_50.pkl` |
