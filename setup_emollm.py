#!/usr/bin/env python3
"""
EmoLLM 情緒模型安裝腳本
========================
架構說明：
  ┌─────────────────────────────────────────────────────────┐
  │  qwen2.5:7b (Ollama)  ← 原本的 AI 居民主要模型          │
  │    負責：對話生成、行程規劃、記憶反思等全部主邏輯        │
  ├─────────────────────────────────────────────────────────┤
  │  EmoLLM (本腳本下載)  ← 獨立的情緒偵測專用模型          │
  │    負責：分析居民行動/對話的情緒狀態（label + intensity）│
  │    輸出：{"label": "焦慮", "intensity": 7, ...}         │
  │    → 情緒結果注入 qwen2.5 的提示詞，影響後續回應        │
  └─────────────────────────────────────────────────────────┘

EmoLLM 是 SmartFlowAI 針對「中文心理諮詢與情緒支持」進行微調的模型系列，
在本系統中只用於情緒偵測，不取代 qwen2.5。

下載來源：ModelScope（中國境內速度更快）
GitHub：https://github.com/SmartFlowAI/EmoLLM

用法：
    python setup_emollm.py              # 互動式安裝
    python setup_emollm.py --model 1    # 直接選擇模型
    python setup_emollm.py --path /已有路徑  # 使用現有模型
"""

import os
import sys
import json
import shutil
import argparse
import subprocess

# ──────────────────────────────────────────────────────────────
# EmoLLM 模型清單（均為 EmoLLM 官方微調版本，非原始底座模型）
# ModelScope ID 已驗證存在（2025-04）
# ──────────────────────────────────────────────────────────────
MODELS = {
    "1": {
        "name":       "EmoLLM V3.0（InternLM2.5-7B 全量微調）★ 最佳品質",
        "desc":       "EmoLLM 最新旗艦版，CPsyCounD 資料集訓練，情緒識別最準確",
        "size":       "約 14 GB",
        "modelscope": "chg0901/EmoLLMV3.0",
        "hf":         "brycewang2018/EmoLLM-mother",
        "local_dir":  "emollm_models/EmoLLM_V3",
    },
    "2": {
        "name":       "EmoLLM（InternLM2-7B 全量微調）",
        "desc":       "穩定版本，InternLM2 全量微調，效果次於 V3",
        "size":       "約 14 GB",
        "modelscope": "ajupyter/EmoLLM_internlm2_7b_full",
        "hf":         None,
        "local_dir":  "emollm_models/EmoLLM_InternLM2_full",
    },
    "3": {
        "name":       "EmoLLM（Qwen2-7B LoRA 微調）",
        "desc":       "Qwen2 系列，與本系統主模型 qwen2.5 同系列架構",
        "size":       "約 14 GB",
        "modelscope": "aJupyter/EmoLLM_Qwen2-7B-Instruct_lora",
        "hf":         None,
        "local_dir":  "emollm_models/EmoLLM_Qwen2",
    },
    "4": {
        "name":       "EmoLLM（Qwen1.5-0.5B 超輕量全量微調）",
        "desc":       "僅需 1 GB，GPU 記憶體不足時使用，速度最快",
        "size":       "約 1 GB",
        "modelscope": "aJupyter/EmoLLM_Qwen1_5-0_5B-Chat_full_sft",
        "hf":         None,
        "local_dir":  "emollm_models/EmoLLM_Qwen1.5_0.5B",
    },
    "5": {
        "name":       "EmoLLM（LLaMA3-8B QLoRA 微調）",
        "desc":       "LLaMA3 版本，英文情境更優，支援中英雙語",
        "size":       "約 16 GB",
        "modelscope": "chg0901/EmoLLM-Llama3-8B-Instruct3.0",
        "hf":         None,
        "local_dir":  "emollm_models/EmoLLM_LLaMA3",
    },
    "6": {
        "name":       "不下載（退回使用 qwen2.5 做情緒偵測）",
        "desc":       "無需額外模型，情緒偵測準確度較低",
        "size":       "0 GB",
        "modelscope": None,
        "hf":         None,
        "local_dir":  None,
    },
}

EMOLLM_GITHUB = "https://github.com/SmartFlowAI/EmoLLM.git"
CONFIG_PATH   = "data/config.json"
REPO_DIR      = "emollm_repo"


# ──────────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────────

def run(cmd, check=True, capture=False):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and result.returncode != 0:
        err = result.stderr.strip() if capture else ""
        raise RuntimeError(f"指令失敗 (exit {result.returncode}): {err}")
    return result


def check_package(pkg):
    result = subprocess.run([sys.executable, "-c", f"import {pkg}"], capture_output=True)
    return result.returncode == 0


def install_package(pip_name):
    print(f"  安裝套件: {pip_name}")
    run(f"{sys.executable} -m pip install {pip_name} -q")


# ──────────────────────────────────────────────────────────────
# 步驟 1：複製 EmoLLM GitHub 倉庫
# ──────────────────────────────────────────────────────────────

def clone_emollm_repo():
    if os.path.isdir(REPO_DIR):
        print(f"[1/4] EmoLLM 倉庫已存在於 ./{REPO_DIR}/，跳過。")
        return
    print(f"[1/4] 複製 EmoLLM 倉庫：{EMOLLM_GITHUB}")
    if not shutil.which("git"):
        print("  未找到 git，跳過複製（不影響情緒偵測功能）")
        return
    run(f"git clone --depth 1 {EMOLLM_GITHUB} {REPO_DIR}")
    print(f"  ✓ 複製完成：./{REPO_DIR}/")


# ──────────────────────────────────────────────────────────────
# 步驟 2：安裝推理套件
# ──────────────────────────────────────────────────────────────

def install_deps():
    print("[2/4] 安裝 EmoLLM 推理依賴...")
    deps = {
        "torch":        "torch",
        "transformers": "transformers>=4.40",
        "accelerate":   "accelerate",
        "sentencepiece":"sentencepiece",
    }
    for pkg, pip_name in deps.items():
        if check_package(pkg):
            print(f"  ✓ {pkg} 已安裝")
        else:
            install_package(pip_name)


# ──────────────────────────────────────────────────────────────
# 步驟 3：下載 EmoLLM 模型權重
# ──────────────────────────────────────────────────────────────

def _check_model_exists(local_dir):
    """檢查目錄內是否已有模型權重檔案"""
    if not os.path.isdir(local_dir):
        return False
    for f in os.listdir(local_dir):
        if f.endswith((".bin", ".safetensors", ".gguf")):
            return True
    # ModelScope 可能把模型放在子目錄
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith((".bin", ".safetensors")):
                return True
    return False


def download_from_modelscope(model_id, local_dir):
    if not check_package("modelscope"):
        install_package("modelscope")
    from modelscope import snapshot_download  # type: ignore
    print(f"  從 ModelScope 下載 EmoLLM：{model_id}")
    path = snapshot_download(model_id, cache_dir=local_dir)
    return path


def download_from_hf(repo_id, local_dir):
    if not check_package("huggingface_hub"):
        install_package("huggingface_hub")
    from huggingface_hub import snapshot_download  # type: ignore
    print(f"  從 HuggingFace 下載 EmoLLM：{repo_id}")
    os.makedirs(local_dir, exist_ok=True)
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return path


def download_model(choice):
    info = MODELS[choice]

    if info["local_dir"] is None:
        print("[3/4] 選擇不下載 EmoLLM，跳過。")
        return None

    local_dir = info["local_dir"]

    if _check_model_exists(local_dir):
        print(f"[3/4] EmoLLM 模型已存在於 {local_dir}，跳過下載。")
        return os.path.abspath(local_dir)

    print(f"[3/4] 下載 EmoLLM 模型：{info['name']}")
    print(f"      ModelScope ID：{info['modelscope']}")
    print(f"      預計大小：{info['size']}")
    os.makedirs(local_dir, exist_ok=True)

    model_path = None

    # 優先 ModelScope
    if info["modelscope"]:
        try:
            model_path = download_from_modelscope(info["modelscope"], local_dir)
            print(f"  ✓ ModelScope 下載成功：{model_path}")
        except Exception as e:
            print(f"  ✗ ModelScope 下載失敗：{e}")

    # 備援 HuggingFace
    if not model_path and info["hf"]:
        try:
            model_path = download_from_hf(info["hf"], local_dir)
            print(f"  ✓ HuggingFace 下載成功：{model_path}")
        except Exception as e:
            print(f"  ✗ HuggingFace 下載失敗：{e}")

    if not model_path:
        ms_id = info["modelscope"] or ""
        hf_id = info["hf"] or ""
        print("\n  ❌ 自動下載失敗，請手動下載後指定路徑：")
        print(f"     python setup_emollm.py --path /你的模型路徑\n")
        if ms_id:
            print(f"     ModelScope：https://modelscope.cn/models/{ms_id}")
        if hf_id:
            print(f"     HuggingFace：https://huggingface.co/{hf_id}")
        return None

    return os.path.abspath(model_path)


# ──────────────────────────────────────────────────────────────
# 步驟 4：更新 data/config.json
# ──────────────────────────────────────────────────────────────

def update_config(model_path):
    print(f"[4/4] 更新 {CONFIG_PATH}...")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    emotion_cfg = config["agent"]["think"].setdefault("emotion", {})
    emotion_cfg["enabled"] = True

    if model_path:
        emotion_cfg["model_path"]   = model_path
        emotion_cfg["ollama_model"] = ""
        print(f"  → EmoLLM 本地路徑：{model_path}")
        print(f"  → 主模型 qwen2.5 不受影響，仍用於居民行為生成")
    else:
        emotion_cfg["model_path"]   = ""
        emotion_cfg["ollama_model"] = ""
        print("  → 情緒偵測將由 qwen2.5 兼做（無 EmoLLM 專用模型）")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"  ✓ {CONFIG_PATH} 已更新")


# ──────────────────────────────────────────────────────────────
# 快速驗證 EmoLLM 輸出格式
# ──────────────────────────────────────────────────────────────

def verify_model(model_path):
    if not model_path:
        return
    print("\n驗證 EmoLLM 載入與情緒 JSON 輸出（可能需要 1-2 分鐘）...")
    test_script = f"""
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = r"{model_path}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"裝置: {{device}}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
).eval()

system = (
    "你是專業情緒分析師，輸出嚴格的 JSON："
    '{{"label":"<情緒>","intensity":<1-10>,"reason":"<15字內>"}}'
    "可選情緒：快樂、悲傷、憤怒、恐懼、厭惡、驚訝、平靜、焦慮、興奮、疲憊"
)
user = "角色：蔡宗陞（咖啡店老闆）\\n請分析：「今天生意很差，又碰到無理取鬧的客人，真的很累」"

messages = [{{"role":"system","content":system}},{{"role":"user","content":user}}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=80, temperature=0.3, do_sample=True, pad_token_id=tokenizer.eos_token_id)

result = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
print(f"EmoLLM 輸出：{{result}}")

m = re.search(r'\\{{[^{{}}]+\\}}', result, re.DOTALL)
if m:
    data = json.loads(m.group())
    print(f"解析結果：label={{data.get('label')}} intensity={{data.get('intensity')}}")
    print("✓ EmoLLM 驗證成功！將作為情緒偵測專用模型使用。")
else:
    print("⚠ JSON 解析失敗，但模型可運行，系統會以備援模式處理輸出。")
"""
    subprocess.run([sys.executable, "-c", test_script])


# ──────────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EmoLLM 情緒模型安裝腳本")
    parser.add_argument("--model", type=str, default=None, help="模型選項 (1/2/3/4)")
    parser.add_argument("--path", type=str, default=None, help="已有 EmoLLM 模型的本地路徑")
    parser.add_argument("--skip-deps", action="store_true", help="跳過套件安裝")
    parser.add_argument("--skip-verify", action="store_true", help="跳過模型驗證")
    args = parser.parse_args()

    print("=" * 64)
    print("  EmoLLM 情緒模型安裝腳本")
    print("  GitHub: https://github.com/SmartFlowAI/EmoLLM")
    print()
    print("  架構：qwen2.5 (主模型) + EmoLLM (情緒偵測) 雙模型並行")
    print("  EmoLLM 只負責情緒分析，不取代 qwen2.5 的居民行為生成")
    print("=" * 64)

    # 直接指定現有路徑
    if args.path:
        abs_path = os.path.abspath(args.path)
        if not os.path.isdir(abs_path):
            print(f"錯誤：路徑不存在：{abs_path}")
            sys.exit(1)
        update_config(abs_path)
        print("\n✅ 完成！重新執行 start.py 即可啟用 EmoLLM 情緒偵測。")
        return

    # 顯示模型選單
    choice = args.model
    if not choice:
        print("\n請選擇 EmoLLM 版本（這些都是 EmoLLM 官方微調模型）：\n")
        for k, v in MODELS.items():
            print(f"  [{k}] {v['name']}")
            print(f"       {v['desc']}")
            if v["size"] != "0 GB":
                print(f"       需要磁碟空間：{v['size']}")
            print()
        choice = input("輸入選項 (1~6) [預設: 1]: ").strip() or "1"

    if choice not in MODELS:
        print(f"無效選項：{choice}（請輸入 1~6）")
        sys.exit(1)

    print(f"\n已選擇：{MODELS[choice]['name']}\n")

    clone_emollm_repo()

    if not args.skip_deps and MODELS[choice]["local_dir"]:
        install_deps()
    else:
        print("[2/4] 跳過套件安裝")

    model_path = download_model(choice)
    update_config(model_path)

    if model_path and not args.skip_verify:
        verify_model(model_path)

    print("\n" + "=" * 64)
    print("✅ EmoLLM 設定完成！")
    print()
    if model_path:
        print(f"  EmoLLM 路徑：{model_path}")
        print()
        print("  運行方式：")
        print("    python start.py --name my_sim --step 10")
        print()
        print("  每位 AI 居民在行動或對話後，EmoLLM 會自動分析情緒，")
        print("  情緒結果回注入 qwen2.5 的提示詞，讓居民根據情緒做出回應。")
    else:
        print("  情緒偵測由 qwen2.5 兼做，準確度略低。")
        print("  如需啟用 EmoLLM，日後可執行：python setup_emollm.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
