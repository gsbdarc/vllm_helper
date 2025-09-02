#!/usr/bin/env python3
# infer_ft_8b.py — exact training-format inference
# Prints running accuracy + UNKNOWN count at checkpoints; saves preds/unknowns.

import os
import sys
import json
import re
import requests
import pandas as pd
from tqdm import tqdm
from time import sleep
from typing import List, Dict, Any

# -----------------------
# Project paths & outputs
# -----------------------
PROJ_DIR = "<project-space>/llm-ft"
TEST_PATH = f"{PROJ_DIR}/data/finetune/test.jsonl"
BASE_PRED_ROOT = f"{PROJ_DIR}/preds"

SAVE_EVERY = 100
TIMEOUT_S = 240
USE_GUIDED_CHOICE = True  # try constrained decoding first; fallback if server rejects

# -----------------------
# Labels & normalization
# -----------------------
LABELS = [
    "AskReddit", "worldnews", "explainlikeimfive", "funny",
    "todayilearned", "science", "sports", "LifeProTips",
    "technology", "books"
]
LABELS_SET_LOWER = {l.lower() for l in LABELS}

def map_to_valid_label(s: str) -> str:
    s_norm = (s or "").strip().strip('"\'` ')
    s_clean = re.sub(r"[^\w]", "", s_norm)
    for lab in LABELS:
        if lab.lower() == s_clean.lower():
            return lab
    return "UNKNOWN"

def normalize_to_label_fallback(text: str) -> str:
    if not text:
        return "UNKNOWN"
    first = text.strip().splitlines()[0] if text.strip() else ""
    m = map_to_valid_label(first)
    if m != "UNKNOWN":
        return m
    raw = text.lower()
    for lab in LABELS:
        if lab.lower() in raw:
            return lab
    return "UNKNOWN"

# -----------------------
# Data I/O
# -----------------------
def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            rows.append({
                "id": o.get("id", i),
                "prompt": o["prompt"],         
                "gold": o.get("completion")
            })
    return rows

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_predictions(out_path: str, pred_list):
    pred_list_sorted = sorted(pred_list, key=lambda x: int(x.get("id", 0)))
    save_json(out_path, pred_list_sorted)

def rebuild_unknowns_file(unknown_path: str, pred_list):
    unknown = [r for r in pred_list if r.get("prediction") == "UNKNOWN"]
    to_save = [
        {
            "id": int(r["id"]),
            "gold": r.get("gold"),
            "prompt": r.get("prompt"),
            "response": r.get("response"),
        }
        for r in unknown
    ]
    save_json(unknown_path, sorted(to_save, key=lambda x: x["id"]))

# -----------------------
# vLLM host/port + adapter
# -----------------------
def read_vllm_host_port():
    scratch = os.environ.get("SCRATCH_BASE")
    if not scratch:
        sys.exit("SCRATCH_BASE environment variable not set.")
    host_file = os.path.join(scratch, "vllm", "host.txt")
    port_file = os.path.join(scratch, "vllm", "port.txt")
    try:
        with open(host_file, "r", encoding="utf-8") as f:
            host = f.read().strip()
        with open(port_file, "r", encoding="utf-8") as f:
            port = f.read().strip()
    except FileNotFoundError as e:
        sys.exit(f"Missing host/port file: {e}")
    return host, port

def resolve_adapter_name() -> str:
    env_name = (os.environ.get("VLLM_LORA_ADAPTER") or "").strip()
    if env_name:
        return env_name
    loras = (os.environ.get("VLLM_LORAS") or "").strip()
    if loras:
        first = loras.split(",")[0].strip()
        if "=" in first:
            return first.split("=", 1)[0].strip()
        if first:
            return first
    return "reddit"

def assert_adapter_listed(base_url: str, adapter_name: str):
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        ids = [m.get("id") for m in (r.json().get("data") or [])]
        if adapter_name in ids:
            print(f"[ok] LoRA '{adapter_name}' is registered as a model id.")
        else:
            print(f"[WARN] LoRA '{adapter_name}' not found in /v1/models. Found: {ids}")
    except Exception as e:
        print(f"[WARN] Could not query /v1/models: {e}")

# -----------------------
# /v1/completions (exact training format) with guided-choice
# -----------------------
def _post_completions(url: str, payload: dict) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()

def call_vllm_completions(base_url: str, model_field: str, prompt: str) -> dict:
    url = f"{base_url}/v1/completions"
    base_payload = {
        "model": model_field,          # adapter-as-model, e.g., "reddit"
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 12,              # enough for 'explainlikeimfive'
        "seed": 1234,
        "repetition_penalty": 1.12,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": ["\n", "\r"],
    }
    if USE_GUIDED_CHOICE:
        payload = dict(base_payload)
        payload["guided_choice"] = LABELS
        try:
            return _post_completions(url, payload)
        except requests.HTTPError as e:
            print(f"[WARN] guided_choice failed ({e}); retrying without constraint.", flush=True)
            return _post_completions(url, base_payload)
    else:
        return _post_completions(url, base_payload)

# -----------------------
# Main
# -----------------------
def main():
    host, port = read_vllm_host_port()
    BASE_MODEL   = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B-Base").strip()
    ADAPTER_NAME = resolve_adapter_name()     # e.g., "reddit"
    MODEL_FIELD  = ADAPTER_NAME               # request the LoRA id as the model

    BASE = f"http://{host}:{port}"
    assert_adapter_listed(BASE, ADAPTER_NAME)

    pred_dir = f"{BASE_PRED_ROOT}/qwen_8b_{ADAPTER_NAME}"
    ensure_dir(pred_dir)
    OUT_JSON     = f"{pred_dir}/predictions_{ADAPTER_NAME}.json"
    UNKNOWN_JSON = f"{pred_dir}/unknown_predictions_{ADAPTER_NAME}.json"
    CM_CSV       = f"{pred_dir}/confusion_{ADAPTER_NAME}.csv"
    METRICS_JSON = f"{pred_dir}/metrics_{ADAPTER_NAME}.json"

    print(f"Using vLLM completions at {host}:{port} | base={BASE_MODEL} | adapter/model={ADAPTER_NAME}")
    print(f"Saving under: {pred_dir}")
    print(f"Loading test set: {TEST_PATH}")

    tests = load_jsonl(TEST_PATH)
    total_examples = len(tests)
    print(f"Loaded {total_examples} examples")

    # Resume: load existing predictions if present
    existing = []
    if os.path.exists(OUT_JSON):
        try:
            existing = json.load(open(OUT_JSON, "r", encoding="utf-8"))
            print(f"Resuming with existing predictions: {len(existing)}")
        except Exception as e:
            print(f"[WARN] Could not read existing predictions: {e}")

    pred_by_id: Dict[int, Dict[str, Any]] = {int(r["id"]): r for r in existing if "id" in r}
    processed_ids = set(pred_by_id.keys())

    total_completion_tokens_run = 0
    total_prompt_tokens_run     = 0
    new_count = 0

    for row in tqdm(tests, total=total_examples, desc="Infer"):
        rid = int(row["id"])
        if rid in processed_ids:
            continue

        prompt = row["prompt"]          # EXACT training prompt
        gold   = row.get("gold")

        try:
            data = call_vllm_completions(BASE, MODEL_FIELD, prompt)
            resp = ((data.get("choices") or [{}])[0].get("text") or "")

            pred = map_to_valid_label(resp)
            if pred == "UNKNOWN":
                pred = normalize_to_label_fallback(resp)

            usage = data.get("usage") or {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if isinstance(pt, int) and pt > 0:
                total_prompt_tokens_run += pt
            if isinstance(ct, int) and ct > 0:
                total_completion_tokens_run += ct

            rec = {
                "id": rid,
                "prompt": prompt,
                "prompt_len": len(prompt),
                "response": resp,
                "response_len": len(resp),
                "gold": gold,
                "prediction": pred,
            }

        except requests.RequestException as e:
            print(f"[WARN] request failed for id={rid}: {e}", flush=True)
            rec = {
                "id": rid,
                "prompt": prompt,
                "prompt_len": len(prompt),
                "response": "",
                "response_len": 0,
                "gold": gold,
                "prediction": "UNKNOWN",
            }
            sleep(0.5)

        pred_by_id[rid] = rec
        new_count += 1

        # Checkpoint: save + print running metrics
        if new_count % SAVE_EVERY == 0:
            all_preds = sorted(pred_by_id.values(), key=lambda x: int(x["id"]))
            save_predictions(OUT_JSON, all_preds)
            rebuild_unknowns_file(UNKNOWN_JSON, all_preds)

            df_ck = pd.DataFrame(all_preds)
            labeled_ck = df_ck[df_ck["gold"].notna()]
            unk_total = int((df_ck["prediction"] == "UNKNOWN").sum())
            if not labeled_ck.empty:
                acc_ck = (labeled_ck["gold"] == labeled_ck["prediction"]).mean()
                print(f"[checkpoint] seen={len(df_ck)} | labeled={len(labeled_ck)} | acc={acc_ck:.4f} | UNKNOWNs={unk_total}", flush=True)
            else:
                print(f"[checkpoint] seen={len(df_ck)} | labeled=0 | UNKNOWNs={unk_total}", flush=True)

    # Final save
    all_preds = sorted(pred_by_id.values(), key=lambda x: int(x["id"]))
    save_predictions(OUT_JSON, all_preds)
    rebuild_unknowns_file(UNKNOWN_JSON, all_preds)
    print(f"Wrote predictions JSON: {OUT_JSON}")
    print(f"Wrote unknowns JSON:    {UNKNOWN_JSON}")

    # ----- Metrics over ALL predictions -----
    df = pd.DataFrame(all_preds)
    labeled = df[df["gold"].notna()].copy()

    if not labeled.empty:
        try:
            from sklearn.metrics import accuracy_score, confusion_matrix
            acc = accuracy_score(labeled["gold"], labeled["prediction"])
            label_order = LABELS + (["UNKNOWN"] if "UNKNOWN" in set(labeled["prediction"]) else [])
            cm = confusion_matrix(labeled["gold"], labeled["prediction"], labels=label_order)
            cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in label_order],
                                    columns=[f"pred_{l}" for l in label_order])
        except Exception:
            acc = float((labeled["gold"] == labeled["prediction"]).mean())
            cm_df = pd.crosstab(labeled["gold"], labeled["prediction"], dropna=False)
            label_order = LABELS + (["UNKNOWN"] if "UNKNOWN" in cm_df.columns else [])
            cm_df = cm_df.reindex(index=label_order, columns=label_order, fill_value=0)
            cm_df.index = [f"true_{l}" for l in cm_df.index]
            cm_df.columns = [f"pred_{l}" for l in cm_df.columns]
    else:
        acc = None
        cm_df = pd.DataFrame()

    if not cm_df.empty:
        cm_df.to_csv(CM_CSV)

    metrics = {
        "base_model_env": os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B").strip(),
        "request_model_field": MODEL_FIELD,
        "adapter_used": True,
        "adapter_name": ADAPTER_NAME,
        "num_examples_total": int(len(df)),
        "num_labeled_total": int(len(labeled)),
        "accuracy_total": acc,
        "labels": LABELS,
        "unknown_total": int((df["prediction"] == "UNKNOWN").sum()),
        "completion_tokens_run": int(total_completion_tokens_run),
        "prompt_tokens_run": int(total_prompt_tokens_run),
        "new_items_processed": new_count,
        "endpoint": "/v1/completions",
        "max_tokens": 12,
        "temperature": 0.0,
        "repetition_penalty": 1.12,
        "guided_choice": USE_GUIDED_CHOICE,
    }
    save_json(METRICS_JSON, metrics)

    # Final prints
    if acc is not None:
        print(f"Final accuracy: {acc:.4f} over {len(labeled)} labeled examples")
    else:
        print("Final accuracy: N/A (no gold labels)")
    print(f"Total UNKNOWNs: {metrics['unknown_total']}")
    if not cm_df.empty:
        print("Confusion matrix saved to:", CM_CSV)
    print(f"Completion tokens (this run): {total_completion_tokens_run}")
    print(f"Prompt tokens (this run):     {total_prompt_tokens_run}")

if __name__ == "__main__":
    main()

