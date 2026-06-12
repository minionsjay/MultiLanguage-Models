# Indonesia (ID) 9-Class Multi-Class Models

> 🇮🇩 Indonesian content moderation | Bahasa Indonesia | 9 labels

## Evolution Timeline

| Step | What | Effect |
|------|------|--------|
| v1 | 8-class Full FT + LoRA | Baselines (F1 0.692 / 0.690) |
| v2 | LSD Label Smoothing Distillation | +Quality check (F1 0.674) |
| v3 | Jury relabeling (300 samples) | Fix ID_SARA/Harassment confusion |
| v4 | NIM data generation (+2,500 ID_SARA) | **ID_SARA 0.42 → 0.62** |
| v5 | LoRA LS 2-seed ensemble + threshold cal | F1 0.709 |
| v6 | Supplement Dangerous + Hate_Speech | **F1 0.732** 🔥 |

## Labels (9)

| # | Label | Description |
|---|-------|-------------|
| 1 | Dangerous_Content | Terrorism, violence, drugs, fraud, CSAM, gore |
| 2 | Harassment | Personal insults, bullying, body shaming |
| 3 | Hate_Speech | Identity-based attacks (non-ID specific) |
| 4 | ID_Blasphemy | Religious blasphemy (Pasal 156a KUHP) + UU ITE defamation |
| 5 | ID_SARA | Suku/Agama/Ras/Antargolongan attacks |
| 6 | Political | NON-Indonesian politics |
| 7 | Sexually_Explicit | Porn, arousal, hookup |
| 8 | other_violation | Other violations |
| 9 | safe | Normal content |

---

## Training Parameters (All LoRA Models)

```python
base_model = "microsoft/mdeberta-v3-base"
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"]
)
learning_rate = 2e-4
epochs = 3
batch_size = 16 (train), 64 (eval)
weight_decay = 0.01
warmup_steps = 100
max_length = 256
precision = FP32 (AMD ROCm)

# Label Smoothing (v5/v6)
loss = CrossEntropyLoss(label_smoothing=0.05)

# LSD Distillation (v2)
T = 3.0, alpha = 0.4, label_smoothing = 0.05
loss = alpha * CE(ls=0.05) + (1-alpha) * KL(teacher/T, student/T) * T^2
```

---

## v6 — Best Model

**Calibrated Macro F1: 0.732 | Accuracy: 0.713 | Rows: 45,753**

```python
model = AutoModelForSequenceClassification.from_pretrained(
    f"{M}/id_data/models_v6/final_s42"
)
tokenizer = AutoTokenizer.from_pretrained(
    f"{M}/id_data/models_v6/final_s42"
)

def predict(text):
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax().item()
    return id2label[pred], torch.softmax(logits,-1)[0][pred].item()
```

### Per-Class Results

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|--------|
| ID_Blasphemy                   | 0.9737 | 0.8810 | 0.9250 | 168 |
| Political                      | 1.0000 | 0.8552 | 0.9219 | 145 |
| safe                           | 0.7873 | 0.7926 | 0.7900 | 2088 |
| Sexually_Explicit              | 0.7507 | 0.7783 | 0.7643 | 654 |
| macro avg                      | 0.7630 | 0.7113 | 0.7325 | 6863 |
| Dangerous_Content              | 0.7140 | 0.7475 | 0.7303 | 982 |
| weighted avg                   | 0.7254 | 0.7134 | 0.7159 | 6863 |
| Hate_Speech                    | 0.8097 | 0.5993 | 0.6888 | 866 |
| ID_SARA                        | 0.6741 | 0.6241 | 0.6481 | 822 |
| other_violation                | 0.6615 | 0.5059 | 0.5733 | 85 |
| Harassment                     | 0.4962 | 0.6182 | 0.5505 | 1053 |

### Thresholds (calibrated)

| Label | Threshold |
|-------|-----------|
| Dangerous_Content              | 0.9455 |
| Harassment                     | 0.6437 |
| Hate_Speech                    | 0.9280 |
| ID_Blasphemy                   | 0.9667 |
| ID_SARA                        | 0.7817 |
| Political                      | 0.6041 |
| Sexually_Explicit              | 0.5387 |
| other_violation                | 0.3590 |
| safe                           | 0.4975 |

---

## v5 — LoRA LS 2-Seed Ensemble

**Calibrated Macro F1: 0.709 | Accuracy: 0.687 | Rows: 42,253**

| Seed | Accuracy | Macro F1 |
|------|----------|----------|
| 42 | 0.6690 | 0.6812 |
| 123 | 0.6764 | 0.6845 |
| Ensemble (soft) | 0.6833 | 0.6952 |
| **Calibrated** | **0.6874** | **0.7085** |

### Per-Class F1 (v5 calibrated)

| Label | F1 |
|-------|------|
| Political                      | 0.9185 |
| ID_Blasphemy                   | 0.9097 |
| safe                           | 0.7939 |
| Sexually_Explicit              | 0.7676 |
| macro avg                      | 0.7085 |
| weighted avg                   | 0.6897 |
| other_violation                | 0.6303 |
| ID_SARA                        | 0.6201 |
| Hate_Speech                    | 0.6034 |
| Dangerous_Content              | 0.5847 |
| Harassment                     | 0.5480 |

---

## v1 — 8-Class Full FT

**Macro F1: 0.692 | Accuracy: 0.634**

| Label | P | R | F1 |
|-------|---|---|------|

---

## v1 — 8-Class LoRA

**Macro F1: 0.690 | Accuracy: 0.643**

## v2 — LSD Label Smoothing Distillation

**Macro F1: 0.674 → calibrated: 0.684**

Note: LSD distillation was 5x slower than standard LoRA LS with minimal gain. Not recommended.

---

## Data Evolution

| Version | Rows | ID_SARA | Dangerous | Hate_Speech | Total Labels |
|---------|------|---------|-----------|-------------|-------------|
| v3 (original) | 38,753 | 2,983 | 4,544 | 3,774 | 9 |
| v5 (ID_SARA+) | 42,253 | 5,483 | 4,544 | 4,274 | 9 |
| v6 (Dangerous+) | 45,753 | 5,483 | 6,544 | 5,774 | 9 |

## Key Insights

1. **NIM data generation works**: ID_SARA +0.20, Dangerous +0.15, Hate_Speech +0.09
2. **LSD not worth it**: 5x slower, +0.005 gain
3. **Ensemble helps**: +0.003 from 2-seed soft voting
4. **Threshold calibration**: +0.013 stable gain
5. **Label smoothing** (ls=0.05) is better than plain CE
