"""
SACF Final — 單一多分支模型載入器
═══════════════════════════════════════════════════════════════════

從 sacf_final.pt 載入 SACFFinalModel（多分支單一模型）。
使用方式：

    from emotion_system.sacf_final_loader import load_sacf_final
    model = load_sacf_final('emotion_system/models/sacf_final.pt', device='cuda:0')
    cls7_logits, cls2_logits, reg = model(input_ids, attention_mask, audio, audio_mask, vision, vision_mask)
    pred = cls7_logits.argmax(dim=-1)   # 0..6 對應 −3..+3 強度
"""
import torch
from pathlib import Path
import sys, importlib.util


def _load_model_class():
    """載入 SACFFinalModel 類別與 MOSIDataset"""
    here = Path(__file__).parent
    src = here / 'training' / 'scaf_final.py'
    spec = importlib.util.spec_from_file_location("scaf_train", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SACFFinalModel, mod.MOSIDataset


def load_sacf_final(ckpt_path: str, device: str = 'cuda:0'):
    """
    從 sacf_final.pt 載入單一多分支模型。
    回傳：SACFFinalModel 實例（可直接 forward）。
    """
    SACFFinalModel, _ = _load_model_class()
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = ck['model_config']
    model = SACFFinalModel(
        lang_model=cfg['lang_model'],
        audio_dim=cfg['audio_dim'],
        vision_dim=cfg['vision_dim'],
        modal_hidden=cfg['modal_hidden'],
        fusion_dim=cfg['fusion_dim'],
        top_k=cfg['top_k'],
        num_classes=cfg['num_classes'],
        dropout=cfg.get('dropout', 0.15),
        num_branches=cfg.get('num_branches', 4),
    )
    model.load_state_dict(ck['model_state_dict'])
    model.to(device); model.eval()
    metrics = ck.get('metrics', {})
    print(f"[SACFFinal] 已載入單一多分支模型")
    print(f"  分支數：{cfg.get('num_branches', 4)}  |  訓練時 Acc-7：{metrics.get('Acc-7', 'N/A')}%")
    return model


# CLI: 在 CMU-MOSI test 上驗證已存的 sacf_final.pt
if __name__ == '__main__':
    import argparse, pickle, numpy as np
    from torch.utils.data import DataLoader
    from scipy.stats import pearsonr
    from sklearn.metrics import f1_score
    from transformers import DebertaV2Tokenizer
    from tqdm import tqdm

    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=str(here / 'models' / 'sacf_final.pt'))
    parser.add_argument('--data', default=str(here / 'data' / 'mosi' / 'unaligned_50.pkl'))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_tta', type=int, default=5)
    args = parser.parse_args()

    print(f"[CLI] 載入 SACFFinal: {args.ckpt}")
    model = load_sacf_final(args.ckpt, device=args.device)
    _, MOSIDataset = _load_model_class()

    print(f"[CLI] 載入 CMU-MOSI test 集...")
    with open(args.data, 'rb') as f: data = pickle.load(f)
    test_labels = np.array(data['test']['regression_labels'])
    test_cls7 = np.clip(np.round(test_labels).astype(int), -3, 3) + 3
    test_cls2 = (test_labels >= 0).astype(int)

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    test_ds = MOSIDataset(data['test'], tokenizer)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # TTA inference
    if args.n_tta > 1:
        model.train()  # MC Dropout
        run_l7, run_l2, run_reg = [], [], []
        for _ in tqdm(range(args.n_tta), desc='TTA'):
            l7s, l2s, regs = [], [], []
            for b in loader:
                ids = b['input_ids'].to(args.device); mask = b['attention_mask'].to(args.device)
                aud = b['audio'].to(args.device); amask = b['audio_mask'].to(args.device)
                vis = b['vision'].to(args.device); vmask = b['vision_mask'].to(args.device)
                with torch.no_grad():
                    l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
                l7s.append(l7.cpu().float().numpy()); l2s.append(l2.cpu().float().numpy()); regs.append(reg.cpu().float().numpy())
            run_l7.append(np.concatenate(l7s)); run_l2.append(np.concatenate(l2s)); run_reg.append(np.concatenate(regs))
        L7 = np.mean(run_l7, axis=0); L2 = np.mean(run_l2, axis=0); R = np.mean(run_reg, axis=0)
    else:
        model.eval()
        l7s, l2s, regs = [], [], []
        with torch.no_grad():
            for b in tqdm(loader, desc='Test'):
                ids = b['input_ids'].to(args.device); mask = b['attention_mask'].to(args.device)
                aud = b['audio'].to(args.device); amask = b['audio_mask'].to(args.device)
                vis = b['vision'].to(args.device); vmask = b['vision_mask'].to(args.device)
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
                l7s.append(l7.cpu().float().numpy()); l2s.append(l2.cpu().float().numpy()); regs.append(reg.cpu().float().numpy())
        L7 = np.concatenate(l7s); L2 = np.concatenate(l2s); R = np.concatenate(regs)

    pred7 = L7.argmax(1); pred2 = L2.argmax(1)
    acc7 = (pred7 == test_cls7).mean()*100
    acc2 = (pred2 == test_cls2).mean()*100
    f1 = f1_score(test_cls2, pred2, average='weighted')*100
    mae = np.abs(R - test_labels).mean()
    corr = pearsonr(R.astype(float), test_labels.astype(float))[0]
    w1 = (np.abs(pred7 - test_cls7) <= 1).mean()*100

    print(f"\n╭─────────────────────────────────────────────────────╮")
    print(f"│  SCAF Final（多分支單一模型）測試結果             │")
    print(f"├─────────────────────────────────────────────────────┤")
    print(f"│  Acc-7    ：{acc7:>6.2f} %                            │")
    print(f"│  Acc-2    ：{acc2:>6.2f} %                            │")
    print(f"│  F1       ：{f1:>6.2f} %                            │")
    print(f"│  MAE      ：{mae:>6.4f}                              │")
    print(f"│  Corr     ：{corr:>6.4f}                              │")
    print(f"│  Within-1 ：{w1:>6.2f} %                            │")
    print(f"╰─────────────────────────────────────────────────────╯")
