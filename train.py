"""
CUDA Fine-tuning 訓練腳本
使用模擬產生的對話數據對本地模型進行 LoRA 微調
"""

import os
import json
import argparse
import glob

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


# ─── 資料集 ────────────────────────────────────────────────────────────────────

class ConversationDataset(Dataset):
    """從模擬結果載入對話數據"""

    def __init__(self, tokenizer, data_path="results/checkpoints", max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self._load(data_path)

    def _load(self, data_path):
        conv_files = glob.glob(os.path.join(data_path, "**/conversation.json"), recursive=True)
        if not conv_files:
            raise FileNotFoundError(f"找不到對話數據：{data_path}/**/conversation.json")

        for path in conv_files:
            with open(path, "r", encoding="utf-8") as f:
                conversations = json.load(f)
            for time_key, events in conversations.items():
                for event in events:
                    for header, chats in event.items():
                        if len(chats) < 2:
                            continue
                        # 將多輪對話轉成 prompt-response 格式
                        for i in range(0, len(chats) - 1, 2):
                            speaker_a, text_a = chats[i]
                            speaker_b, text_b = chats[i + 1]
                            self.samples.append({
                                "prompt": f"{speaker_a}: {text_a}",
                                "response": f"{speaker_b}: {text_b}",
                            })

        print(f"載入 {len(self.samples)} 筆訓練樣本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        response = sample["response"]

        full_text = f"User: {prompt}\nAssistant: {response}{self.tokenizer.eos_token}"
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # 計算 prompt 長度，prompt 部分的 label 設為 -100（不計算 loss）
        prompt_text = f"User: {prompt}\nAssistant: "
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_ids["input_ids"])

        labels = tokenized["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len

        tokenized["labels"] = labels
        return tokenized


# ─── 主程式 ────────────────────────────────────────────────────────────────────

def main(args):
    # 裝置檢查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告：未偵測到 CUDA，訓練將使用 CPU（速度較慢）")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"使用 GPU: {gpu_name}（顯存: {gpu_mem:.1f} GB）")

    # 載入 tokenizer 和模型
    print(f"載入模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    # 套用 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(",") if args.lora_targets else None,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 準備資料集
    dataset = ConversationDataset(tokenizer, args.data_path, args.max_length)

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=(device == "cuda"),
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        warmup_ratio=0.05,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    # 開始訓練
    print("開始訓練...")
    trainer.train()

    # 儲存模型
    print(f"儲存模型至 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("訓練完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA LoRA 微調訓練")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace 模型路徑或名稱（例如 Qwen/Qwen2-7B）")
    parser.add_argument("--data_path", type=str, default="results/checkpoints",
                        help="模擬對話數據路徑")
    parser.add_argument("--output_dir", type=str, default="results/finetuned_model",
                        help="訓練後模型輸出路徑")
    parser.add_argument("--epochs", type=int, default=3, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=2, help="每步 batch 大小")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累積步數")
    parser.add_argument("--lr", type=float, default=2e-4, help="學習率")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列長度")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_targets", type=str, default="",
                        help="LoRA 目標層，逗號分隔（例如 q_proj,v_proj），留空自動偵測")
    parser.add_argument("--save_steps", type=int, default=100, help="每幾步儲存一次 checkpoint")
    args = parser.parse_args()
    main(args)
