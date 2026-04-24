# SCAF v15 系统流程图

## 1️⃣ 整体架构流程

```mermaid
graph TB
    Start([开始训练]) --> LoadData[加载 MOSI 数据集]
    LoadData --> DataSplit{数据分割}
    DataSplit --> Train[训练集 1284]
    DataSplit --> Valid[验证集 229]
    DataSplit --> Test[测试集 686]

    Train --> DataProcess[数据预处理模块]
    Valid --> DataProcess
    Test --> DataProcess

    DataProcess --> Model[HybridModel 主模型]

    Model --> Output{三路输出}
    Output --> CLS7[7类情感分类<br/>-3到+3]
    Output --> CLS2[2类情感分类<br/>正面/负面]
    Output --> REG[情感强度回归<br/>连续值-3到3]

    CLS7 --> Loss[混合损失函数]
    CLS2 --> Loss
    REG --> Loss

    Loss --> Backward[反向传播]
    Backward --> Optimize[优化器更新]
    Optimize --> Schedule[学习率调度]
    Schedule --> Unfreeze{需要解冻?}

    Unfreeze -->|是| RebuildOpt[重建优化器]
    Unfreeze -->|否| NextEpoch{继续训练?}
    RebuildOpt --> NextEpoch

    NextEpoch -->|未达70 epochs| Model
    NextEpoch -->|完成| Evaluate[测试集评估]
    Evaluate --> End([训练完成])

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Model fill:#87CEEB
    style Loss fill:#FFD700
```

## 2️⃣ 数据预处理详细流程

```mermaid
graph LR
    Input([原始输入]) --> Text[文本数据]
    Input --> Audio[音频特征<br/>5维 x 时序]
    Input --> Vision[视觉特征<br/>20维 x 时序]

    Text --> AddPrompt[添加任务提示词<br/>TASK_PROMPT]
    AddPrompt --> Tokenize[DeBERTa Tokenizer<br/>最大长度80]
    Tokenize --> TextOut[input_ids<br/>attention_mask]

    Audio --> AudioMask[生成音频mask<br/>标记有效帧]
    AudioMask --> AudioOut[audio tensor<br/>audio_mask]

    Vision --> VisionMask[生成视觉mask<br/>标记有效帧]
    VisionMask --> VisionOut[vision tensor<br/>vision_mask]

    Input --> Label[标签数据]
    Label --> GenLabel[生成三种标签]
    GenLabel --> Label7[cls7: 7类标签]
    GenLabel --> Label2[cls2: 2类标签]
    GenLabel --> LabelR[reg: 回归标签]

    TextOut --> Batch[组成Batch]
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

## 3️⃣ 模型前向传播核心流程

```mermaid
graph TB
    Input([输入数据]) --> Branch{三路并行处理}

    %% 文本分支
    Branch -->|文本| DeBERTa[DeBERTa-v3-large<br/>24层 Transformer<br/>1024维]
    DeBERTa --> Hidden[隐藏状态<br/>Batch x 80 x 1024]
    Hidden --> PolarityAttn[Polarity Enhanced<br/>Attention]
    PolarityAttn --> Gate[门控机制<br/>0.75*h + 0.25*h*g]
    Gate --> LangPool[池化<br/>Batch x 1024]

    %% 音频分支
    Branch -->|音频| AudioProj[投影层<br/>5维 → 256维]
    AudioProj --> AudioPos[位置编码]
    AudioPos --> AudioTrans[Transformer编码器<br/>2层 4头]
    AudioTrans --> AudioAttnPool[Attention Pooling]
    AudioAttnPool --> AudioOut[Batch x 256]

    %% 视觉分支
    Branch -->|视觉| VisionProj[投影层<br/>20维 → 256维]
    VisionProj --> VisionPos[位置编码]
    VisionPos --> VisionTrans[Transformer编码器<br/>2层 4头]
    VisionTrans --> VisionAttnPool[Attention Pooling]
    VisionAttnPool --> VisionOut[Batch x 256]

    %% 跨模态融合
    LangPool --> CrossModal[跨模态注意力]
    AudioOut --> CrossModal
    VisionOut --> CrossModal

    CrossModal --> Align[模态对齐<br/>256维 → 1024维]
    Align --> AttnScore[注意力计算<br/>scale dot-product]
    AttnScore --> AttnWeight[softmax权重]
    AttnWeight --> Attended[加权融合]
    Attended --> GateCtrl[门控调节]
    GateCtrl --> Residual[残差连接]
    Residual --> FusedFeat[融合特征<br/>Batch x 1024]

    %% 共享层和预测头
    FusedFeat --> Shared[共享特征层<br/>1024 → 512<br/>LayerNorm + GELU]

    Shared --> Split{三个预测头}
    Split --> Head7[CLS7 Head<br/>Linear 512→7]
    Split --> Head2[CLS2 Head<br/>Linear 512→2]
    Split --> HeadReg[REG Head<br/>512→256→1<br/>Tanh*3.0]

    Head7 --> Out7([7类logits])
    Head2 --> Out2([2类logits])
    HeadReg --> OutReg([回归值])

    style DeBERTa fill:#87CEEB
    style AudioTrans fill:#98FB98
    style VisionTrans fill:#DDA0DD
    style CrossModal fill:#FFD700
    style Shared fill:#FFA07A
```

## 4️⃣ 跨模态注意力机制详解

```mermaid
graph TB
    subgraph Input [输入特征]
        Lang[语言特征 xl<br/>1024维]
        Aud[音频特征 xa<br/>256维]
        Vis[视觉特征 xv<br/>256维]
    end

    subgraph Alignment [步骤1: 模态对齐]
        Aud --> MapA[Linear Audio→Lang<br/>256→1024]
        Vis --> MapV[Linear Vision→Lang<br/>256→1024]
        MapA --> AudAlign[xa_mapped<br/>1024维]
        MapV --> VisAlign[xv_mapped<br/>1024维]
    end

    subgraph Attention [步骤2: 注意力计算]
        Lang --> Query[Query: xl]
        AudAlign --> Stack[Stack KV]
        VisAlign --> Stack
        Stack --> KV[Key-Value<br/>Batch x 2 x 1024]

        Query --> MatMul[xl @ KV^T]
        KV --> MatMul
        MatMul --> Scale[除以 √1024]
        Scale --> Softmax[Softmax<br/>注意力权重]
        Softmax --> WeightedSum[加权求和]
        KV --> WeightedSum
        WeightedSum --> Attended[attended<br/>Batch x 1024]
    end

    subgraph Fusion [步骤3: 门控融合]
        Lang --> Add1[残差: xl + attended]
        Attended --> Add1
        Add1 --> FFN[前馈网络<br/>1024→512→1024]
        FFN --> Enhanced[x_enhanced]

        Lang --> Concat[Concat]
        Enhanced --> Concat
        Concat --> GateCalc[Linear + Sigmoid<br/>2048→1]
        GateCalc --> GateW[gate_weight ∈ 0,1]

        Enhanced --> Multiply[x * gate_weight]
        GateW --> Multiply
        Multiply --> Gated[gated_x]

        Lang --> ResAdd[xl + dropout]
        Gated --> ResAdd
        ResAdd --> Norm[LayerNorm]
        Norm --> Output[输出特征<br/>1024维]
    end

    style Lang fill:#E6F3FF
    style Aud fill:#E6FFE6
    style Vis fill:#FFE6F3
    style Output fill:#FFE6B3
```

## 5️⃣ 训练策略时间线

```mermaid
gantt
    title SCAF v15 训练70个Epochs的策略
    dateFormat X
    axisFormat %s

    section 语言模型解冻
    冻结前6层 (训练18层)    :freeze6, 0, 23
    冻结前3层 (训练21层)    :freeze3, 23, 47
    全部解冻 (训练24层)     :freeze0, 47, 70

    section 学习率调度
    Warmup (线性增长)       :warm, 0, 7
    Cosine衰减              :cosine, 7, 70

    section 训练阶段
    初期快速学习            :crit, 0, 15
    中期稳定优化            :active, 15, 50
    后期精细调优            :50, 70
```

## 6️⃣ 损失函数计算流程

```mermaid
graph TB
    Pred([模型预测]) --> Split{三路输出}
    Label([真实标签]) --> SplitL{三种标签}

    Split --> Logits7[7类logits]
    Split --> Logits2[2类logits]
    Split --> RegOut[回归预测值]

    SplitL --> Label7[cls7标签]
    SplitL --> Label2[cls2标签]
    SplitL --> LabelR[reg标签]

    Logits7 --> FocalLoss[Focal Loss]
    Label7 --> FocalLoss
    FocalLoss --> W7[类别权重]
    W7 --> CE7[交叉熵]
    CE7 --> PT[预测概率 pt]
    PT --> Modulate[调制因子<br/>1-pt^2.0]
    Modulate --> Smooth7[标签平滑 0.1]
    Smooth7 --> Loss7[L_cls7]

    Logits2 --> CE2[交叉熵<br/>标签平滑0.05]
    Label2 --> CE2
    CE2 --> Loss2[L_cls2]

    RegOut --> SmoothL1[Smooth L1 Loss]
    LabelR --> SmoothL1
    SmoothL1 --> LossR[L_reg]

    Loss7 --> Combine[加权组合]
    Loss2 --> Combine
    LossR --> Combine

    Combine --> Formula[Total = 3.0*L_cls7<br/>+ 0.5*L_cls2<br/>+ 0.3*L_reg]
    Formula --> TotalLoss([总损失])

    TotalLoss --> Backward[反向传播]
    Backward --> Clip[梯度裁剪<br/>max_norm=1.0]
    Clip --> Update[优化器更新]

    style FocalLoss fill:#FFB6C1
    style Formula fill:#FFD700
    style TotalLoss fill:#FF6B6B
```

## 7️⃣ 渐进式解冻决策树

```mermaid
graph TB
    Start([当前Epoch]) --> Check{Epoch < 23?}

    Check -->|是| Stage1[阶段1: 前1/3]
    Check -->|否| Check2{Epoch < 47?}

    Stage1 --> Freeze6[冻结前6层<br/>训练后18层]
    Freeze6 --> Params1[可训练参数<br/>约200M]

    Check2 -->|是| Stage2[阶段2: 中1/3]
    Check2 -->|否| Stage3[阶段3: 后1/3]

    Stage2 --> Freeze3[冻结前3层<br/>训练后21层]
    Freeze3 --> Params2[可训练参数<br/>约250M]

    Stage3 --> Freeze0[全部解冻<br/>训练全部24层]
    Freeze0 --> Params3[可训练参数<br/>约300M]

    Params1 --> Changed{参数状态<br/>改变?}
    Params2 --> Changed
    Params3 --> Changed

    Changed -->|是| Rebuild[重建优化器<br/>更新参数组]
    Changed -->|否| Continue[继续训练]

    Rebuild --> Update[更新lang_params列表<br/>重新初始化AdamW]
    Update --> Continue

    Continue --> End([继续下一轮])

    style Stage1 fill:#FFE6E6
    style Stage2 fill:#FFF4E6
    style Stage3 fill:#E6FFE6
    style Rebuild fill:#FFD700
```

## 8️⃣ 评估指标计算流程

```mermaid
graph LR
    Pred([模型预测]) --> P7[7类预测]
    Pred --> P2[2类预测]
    Pred --> PR[回归预测]

    Label([真实标签]) --> L7[7类标签]
    Label --> L2[2类标签]
    Label --> LR[回归标签]

    P7 --> Acc7Calc[预测==标签<br/>求平均]
    L7 --> Acc7Calc
    Acc7Calc --> Acc7Metric[Acc7 %]

    P2 --> Acc2Calc[预测==标签<br/>求平均]
    L2 --> Acc2Calc
    Acc2Calc --> Acc2Metric[Acc2 %]

    P2 --> F1Calc[F1-Score<br/>weighted]
    L2 --> F1Calc
    F1Calc --> F1Metric[F1 %]

    PR --> MAECalc[平均绝对误差<br/>mean abs]
    LR --> MAECalc
    MAECalc --> MAEMetric[MAE]

    PR --> CorrCalc[皮尔逊相关<br/>pearsonr]
    LR --> CorrCalc
    CorrCalc --> CorrMetric[Corr]

    Acc7Metric --> Target1{> 51%?}
    Acc2Metric --> Target2{> 85%?}
    MAEMetric --> Target3{< 0.7?}
    CorrMetric --> Target4{> 0.8?}

    Target1 --> Eval{达到目标?}
    Target2 --> Eval
    Target3 --> Eval
    Target4 --> Eval

    Eval -->|全部达标| Success([训练成功<br/>达到SOTA])
    Eval -->|未达标| Continue([继续训练<br/>或调整策略])

    style Success fill:#90EE90
    style Continue fill:#FFD700
```

## 📊 关键超参数配置表

| 组件 | 参数 | v15值 | 说明 |
|------|------|-------|------|
| **数据** | Batch Size | 8 | 小batch更稳定 |
| | Max Text Len | 80 | DeBERTa输入长度 |
| **模型** | Audio Dim | 5 | COVAREP特征 |
| | Vision Dim | 20 | Facet特征 |
| | Modal Hidden | 256 | 非语言编码维度 |
| | Fusion Dim | 512 | 融合层维度 |
| | Dropout | 0.2 | 正则化 |
| **优化** | Lang LR | 5e-6 | 语言模型学习率 |
| | Other LR | 1e-4 | 其他模块学习率 |
| | Weight Decay | 1e-2 | L2正则 |
| | Warmup Ratio | 0.1 | 10%步数预热 |
| | Epochs | 70 | 训练轮数 |
| **损失** | Alpha (cls7) | 3.0 | 7类权重 |
| | Beta (cls2) | 0.5 | 2类权重 |
| | Gamma (reg) | 0.3 | 回归权重 |
| | Focal Gamma | 2.0 | Focal Loss参数 |

## 🎯 训练目标对比

```
Epoch 1  预期: >= 20%   (v10仅4.37%, old达15.72%)
Epoch 10 预期: >= 40%
Epoch 70 预期: >= 51%   (MGT SOTA: 55.6%)
```
