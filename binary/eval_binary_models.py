#!/usr/bin/env python3
"""
Binary model evaluation across all 8 countries × 4 methods.
Loads random samples, runs all models, saves detailed results.
"""
import json, os, sys, time, random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

# Offline mode
os.environ['HF_HUB_OFFLINE'] = '1'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ============================================================
# Config
# ============================================================
BASE = "/home/ninini/Agents/Colloation_data/Training"
MODEL_REGISTRY = {
    "Full FT": f"{BASE}/binary_8countries",
    "LoRA": f"{BASE}/binary_8countries",
    "Multi-LoRA": f"{BASE}/multi_lora",
    "Adapter": f"{BASE}/adapter",
}

COUNTRIES = {
    'SA': 'Saudi Arabia', 'BR': 'Brazil', 'MX': 'Mexico',
    'ID': 'Indonesia', 'SG': 'Singapore', 'TH': 'Thailand',
    'TR': 'Turkey', 'ZA': 'South Africa',
}

# Data sources for sampling
DATA_SOURCES = {
    'SA': '/home/ninini/Datasets_Nine_country/Saudi-Arabia/all_datasets_with_labels_dedup.csv',
    'BR': '/home/ninini/Datasets_Nine_country/Brazil/all_datasets_with_labels_dedup.csv',
    'MX': '/home/ninini/Datasets_Nine_country/Mexico/all_datasets_with_labels_dedup.csv',
    'ID': '/home/ninini/Datasets_Nine_country/Indonesia/all_datasets_with_labels_dedup.csv',
    'SG': '/home/ninini/Datasets_Nine_country/Singapore/all_datasets_with_labels_dedup.csv',
    'TH': '/home/ninini/Datasets_Nine_country/Thailand/all_datasets_with_labels_dedup.csv',
    'TR': '/home/ninini/Datasets_Nine_country/Turkiye/all_datasets_with_labels_dedup.csv',
    'ZA': '/home/ninini/Datasets_Nine_country/South-Africa/all_datasets_with_labels_dedup.csv',
}

OUT_DIR = f"{BASE}/model_reports/binary/eval_results"
os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLES = 200  # per country

# ============================================================
# Sample data loading
# ============================================================
def load_samples(country_code, n=N_SAMPLES):
    """Load balanced random samples from country data."""
    path = DATA_SOURCES.get(country_code)
    if not path or not os.path.exists(path):
        print(f"  ⚠️ No data for {country_code}")
        return None

    df = pd.read_csv(path)
    if 'label' not in df.columns:
        return None

    df = df[['text', 'label']].dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    # Balance: take min(len(safe), len(viol), n//2) from each
    safe = df[df['label'] == 0]
    viol = df[df['label'] >= 1]

    n_per = min(n // 2, len(safe), len(viol))
    if n_per < 5:
        return df.sample(n=min(n, len(df)), random_state=42)

    sampled = pd.concat([
        safe.sample(n=n_per, random_state=42),
        viol.sample(n=n_per, random_state=42)
    ], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled

# ============================================================
# Model loading
# ============================================================
def load_fullft(country_code):
    path = f"{BASE}/binary_8countries/{country_code}_fullft/final_model"
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

def load_lora(country_code):
    """Load LoRA via PeftModel, do inference directly (no merge)."""
    try:
        from peft import PeftModel
        path = f"{BASE}/binary_8countries/{country_code}_lora/final_model"
        base = AutoModelForSequenceClassification.from_pretrained(
            "/home/ninini/.cache/huggingface/hub/models--microsoft--mdeberta-v3-base/snapshots/a0484667b22365f84929a935b5e50a51f71f159d",
            num_labels=2, local_files_only=True, ignore_mismatched_sizes=True)
        # Use PEFT model directly - DON'T merge
        model = PeftModel.from_pretrained(base, path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model.to(DEVICE)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"    LoRA load error: {e}")
        return None, None

def load_multilora(country_code):
    """Load Multi-LoRA via PeftModel directly."""
    try:
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            "/home/ninini/.cache/huggingface/hub/models--microsoft--mdeberta-v3-base/snapshots/a0484667b22365f84929a935b5e50a51f71f159d",
            num_labels=2, local_files_only=True, ignore_mismatched_sizes=True)
        path = f"{BASE}/multi_lora/{country_code}/lora_weights"
        if not os.path.exists(path):
            return None, None
        model = PeftModel.from_pretrained(base, path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model.to(DEVICE)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"    Multi-LoRA load error: {e}")
        return None, None

def load_adapter(country_code):
    """Load Adapter model (uses adapters library)."""
    path = f"{BASE}/adapter/{country_code}/adapter_weights"
    if not os.path.exists(path):
        return None, None
    try:
        # Adapter models were saved with adapter-transformers library
        # May need specific loading logic
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model.to(DEVICE)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"    Adapter load error ({country_code}): {str(e)[:80]}")
        return None, None

LOADERS = {
    "Full FT": load_fullft,
    "LoRA": load_lora,
    "Multi-LoRA": load_multilora,
    "Adapter": load_adapter,
}

# ============================================================
# Prediction
# ============================================================
@torch.no_grad()
def predict_batch(model, tokenizer, texts, batch_size=32):
    """Batch predict and return probabilities and predictions."""
    all_probs = []
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=256, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_preds)

# ============================================================
# Main evaluation
# ============================================================
all_results = {}
model_load_times = {}

print("="*70)
print("BINARY MODEL EVALUATION — 8 Countries × 4 Methods")
print("="*70)

for method_name, loader_fn in LOADERS.items():
    print(f"\n{'='*70}")
    print(f"METHOD: {method_name}")
    print(f"{'='*70}")

    method_results = {}

    for code, name in COUNTRIES.items():
        print(f"\n  Country: {name} ({code})")

        # Load data
        samples = load_samples(code)
        if samples is None or len(samples) < 10:
            print(f"    ❌ Insufficient data")
            continue

        true_labels = samples['label'].apply(lambda x: 0 if x == 0 else 1).values
        texts = samples['text'].tolist()

        print(f"    Samples: {len(samples)} (safe={sum(true_labels==0)}, viol={sum(true_labels==1)})")

        # Load model
        t0 = time.time()
        model, tokenizer = loader_fn(code)
        load_time = time.time() - t0

        if model is None:
            print(f"    ❌ Model not available")
            continue

        model_load_times[f"{method_name}_{code}"] = load_time
        print(f"    Load time: {load_time:.1f}s")

        # Predict
        t0 = time.time()
        probs, preds = predict_batch(model, tokenizer, texts)
        pred_time = time.time() - t0
        print(f"    Predict time: {pred_time:.2f}s ({pred_time/len(texts)*1000:.0f}ms/sample)")

        # Metrics
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        p, r, f1, s = precision_recall_fscore_support(true_labels, preds, labels=[0, 1], zero_division=0)

        # Save per-sample predictions
        sample_results = []
        for i in range(len(texts)):
            sample_results.append({
                'text': texts[i][:200],
                'true_label': int(true_labels[i]),
                'pred_label': int(preds[i]),
                'prob_safe': float(probs[i][0]),
                'prob_violation': float(probs[i][1]),
                'correct': bool(true_labels[i] == preds[i]),
            })

        method_results[code] = {
            'country': name,
            'n_samples': len(texts),
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'macro_precision': float(np.mean(p)),
            'macro_recall': float(np.mean(r)),
            'clean': {'precision': float(p[0]), 'recall': float(r[0]), 'f1': float(f1[0]), 'support': int(s[0])},
            'violation': {'precision': float(p[1]), 'recall': float(r[1]), 'f1': float(f1[1]), 'support': int(s[1])},
            'load_time': load_time,
            'pred_time': pred_time,
            'pred_ms_per_sample': pred_time / len(texts) * 1000,
            'samples': sample_results,
        }

        print(f"    Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Viol R: {r[1]:.4f} | Viol P: {p[1]:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results[method_name] = method_results

# ============================================================
# Cross-Comparison: Generate per-country summary
# ============================================================
print(f"\n{'='*70}")
print("GENERATING SUMMARY REPORTS")
print(f"{'='*70}")

ts = time.strftime("%Y%m%d_%H%M%S")

# 1. Full JSON results
with open(f"{OUT_DIR}/eval_results_{ts}.json", 'w') as f:
    json.dump({
        'config': {'n_samples_per_country': N_SAMPLES, 'device': DEVICE},
        'model_load_times': model_load_times,
        'results': all_results,
    }, f, indent=2, ensure_ascii=False)
print(f"✅ JSON: eval_results_{ts}.json")

# 2. Per-country comparison CSV
rows = []
for code, name in COUNTRIES.items():
    row = {'country': code, 'name': name}
    for method in ["Full FT", "LoRA", "Multi-LoRA", "Adapter"]:
        if method in all_results and code in all_results[method]:
            m = all_results[method][code]
            for metric in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']:
                row[f"{method}_{metric}"] = m[metric]
            for cls_name, cls_key in [('clean', 'clean'), ('violation', 'violation')]:
                for mtr in ['precision', 'recall', 'f1']:
                    row[f"{method}_{cls_key}_{mtr}"] = m[cls_key][mtr]
    rows.append(row)

df_comp = pd.DataFrame(rows)
df_comp.to_csv(f"{OUT_DIR}/country_comparison_{ts}.csv", index=False)
print(f"✅ CSV: country_comparison_{ts}.csv")

# 3. Per-country detailed Markdown
for code, name in COUNTRIES.items():
    md = f"# {code} ({name}) — Binary Model Evaluation\n\n"
    md += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')} | **Samples:** {N_SAMPLES} per country\n\n"

    md += "## Overall Results\n\n"
    md += "| Method | Acc | Macro F1 | Clean P | Clean R | Clean F1 | Viol P | Viol R | Viol F1 |\n"
    md += "|--------|-----|----------|---------|---------|----------|--------|--------|----------|\n"
    for method in ["Full FT", "LoRA", "Multi-LoRA", "Adapter"]:
        if method in all_results and code in all_results[method]:
            m = all_results[method][code]
            c = m['clean']; v = m['violation']
            md += f"| {method} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |\n"

    md += "\n## Misclassified Examples\n\n"
    # Show top errors for each method
    for method in ["Full FT", "LoRA"]:
        if method in all_results and code in all_results[method]:
            errors = [s for s in all_results[method][code].get('samples', []) if not s['correct']]
            false_pos = [s for s in errors if s['pred_label'] == 1 and s['true_label'] == 0]
            false_neg = [s for s in errors if s['pred_label'] == 0 and s['true_label'] == 1]

            md += f"### {method}\n"
            if false_neg:
                md += f"\n**False Negatives** (violation → safe, DANGEROUS):\n\n"
                for s in false_neg[:5]:
                    md += f"- ❌ `{s['text'][:100]}...`\n"
            if false_pos:
                md += f"\n**False Positives** (safe → violation):\n\n"
                for s in false_pos[:5]:
                    md += f"- ⚠️ `{s['text'][:100]}...`\n"

    with open(f"{OUT_DIR}/{code}_eval_{ts}.md", 'w') as f:
        f.write(md)
    print(f"✅ {code}_eval_{ts}.md")

# 4. Global comparison table
global_md = f"""# Binary Model Global Comparison

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')} | **Samples:** {N_SAMPLES}/country

## All Results Summary

| Country | Full FT Acc | Full FT F1 | LoRA Acc | LoRA F1 | MLoRA Acc | MLoRA F1 | Adptr Acc | Adptr F1 |
|---------|------------|------------|----------|---------|-----------|----------|-----------|----------|
"""
for code, name in COUNTRIES.items():
    row = f"| {code}"
    for m in ["Full FT", "LoRA", "Multi-LoRA", "Adapter"]:
        if m in all_results and code in all_results[m]:
            r = all_results[m][code]
            row += f" | {r['accuracy']:.4f} | {r['macro_f1']:.4f}"
        else:
            row += " | - | -"
    global_md += row + " |\n"

global_md += """
## Violation Recall Ranking (most important)

| Rank | Country | Method | Viol Recall | Viol Precision | Viol F1 |
|------|--------|--------|-------------|---------------|--------|
"""
viol_all = []
for m in all_results:
    for code in all_results[m]:
        v = all_results[m][code]['violation']
        viol_all.append((v['recall'], code, m, v['precision'], v['f1']))
viol_all.sort(reverse=True)

for i, (rec, code, method, prec, f1) in enumerate(viol_all, 1):
    global_md += f"| {i} | {code} | {method} | {rec:.4f} | {prec:.4f} | {f1:.4f} |\n"

global_md += f"""
## Files

- JSON: `eval_results_{ts}.json` (all metrics + per-sample predictions)
- CSV: `country_comparison_{ts}.csv` (wide format for analysis)
- Per-country MD: `SA/BR/MX/..._eval_{ts}.md` (8 files, one per country)
- This README: `README_{ts}.md`
"""

with open(f"{OUT_DIR}/README_{ts}.md", 'w') as f:
    f.write(global_md)
print(f"✅ README_{ts}.md")

print(f"\n{'='*70}")
print(f"EVALUATION COMPLETE")
print(f"Results saved to: {OUT_DIR}")
print(f"Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
print(f"{'='*70}")
