# Singapore (SG) Multi-Class Model

> 🇸🇬 Singapore content moderation | English/Singlish | 9 labels

## Model Overview

| Attribute | Value |
|-----------|-------|
| Country | Singapore (SG) |
| Languages | English, Singlish (Hokkien/Malay/Tamil code-switching) |
| Task | 9-class content moderation |
| Method | LoRA Label Smoothing + Threshold Calibration |
| Base Model | microsoft/mdeberta-v3-base |
| Model Path | `Training/sg_data/models/final_model` |
| Data Rows | 25,543 |
| Labels | 9 |

## Labels (9)

| # | Label | Description |
|---|-------|-------------|
| 1 | Dangerous_Content | Terrorism, violence, drugs, fraud, CSAM |
| 2 | Harassment | Personal insults, bullying |
| 3 | Hate_Speech | Identity-based attacks |
| 4 | Sexually_Explicit_Information | Porn, arousal, hookup |
| 5 | Politically_Sensitive_Topics | NON-SG politics |
| 6 | Cybersecurity_Malware | Malware, hacking, phishing |
| 7 | SG_Racial_Religious_Harmony | Racial/religious disharmony (Sedition Act, MRHA) |
| 8 | SG_Vulgarity_Singlish | Heavy Hokkien vulgarities (chee bai, kan ni na, etc.) |
| 9 | safe | Normal SG life |

## Training Parameters

```python
base_model = "microsoft/mdeberta-v3-base"
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"]
)
learning_rate = 2e-4
epochs = 3
batch_size = 16 (train), 64 (eval)
label_smoothing = 0.05
weight_decay = 0.01
warmup_steps = 100
seed = 42
```

## Evaluation Results

**Seed Macro F1: 0.6686 | Calibrated Macro F1: 0.7043**

### Per-Class

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|--------|
| Politically_Sensitive_Topics        | 0.9897 | 0.9970 | 0.9934 | 675 |
| Hate_Speech                         | 0.9533 | 0.9985 | 0.9754 | 675 |
| safe                                | 0.8646 | 0.8278 | 0.8458 | 1173 |
| weighted avg                        | 0.8204 | 0.8155 | 0.8168 | 3832 |
| Dangerous_Content                   | 0.7716 | 0.7908 | 0.7811 | 282 |
| Sexually_Explicit_Information       | 0.7626 | 0.7076 | 0.7341 | 277 |
| Harassment                          | 0.7594 | 0.6827 | 0.7190 | 208 |
| macro avg                           | 0.7082 | 0.7048 | 0.7043 | 3832 |
| SG_Racial_Religious_Harmony         | 0.5163 | 0.4922 | 0.5040 | 193 |
| SG_Vulgarity_Singlish               | 0.3626 | 0.5130 | 0.4249 | 193 |
| Cybersecurity_Malware               | 0.3939 | 0.3333 | 0.3611 | 156 |

### Thresholds

| Label | Threshold |
|-------|-----------|
| Cybersecurity_Malware               | 0.2418 |
| Dangerous_Content                   | 0.9367 |
| Harassment                          | 0.9473 |
| Hate_Speech                         | 0.7076 |
| Politically_Sensitive_Topics        | 0.8693 |
| SG_Racial_Religious_Harmony         | 0.4416 |
| SG_Vulgarity_Singlish               | 0.2524 |
| Sexually_Explicit_Information       | 0.5033 |
| safe                                | 0.5016 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/sg_data/models/final_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/sg_data/models/final_model"
)

labels = ["Dangerous_Content", "Harassment", "Hate_Speech",
          "Sexually_Explicit_Information", "Politically_Sensitive_Topics",
          "Cybersecurity_Malware", "SG_Racial_Religious_Harmony",
          "SG_Vulgarity_Singlish", "safe"]

def predict(text):
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, -1)
        pred = logits.argmax().item()
    return labels[pred], probs[0][pred].item()

# Example
label, conf = predict("Wah lao this CB driver almost bang my car, damn bodoh!")
print(f"{label}: {conf:.3f}")
```

## Data Source

- Primary: `data_v2/train_SG.csv` (12,815 rows, multi-class labels)
- Supplement: `Datasets_Nine_country/Singapore/` (5,000 safe samples)
- Generated: NIM API (Hate_Speech 4,500 + Politically_Sensitive_Topics 4,500)

## Issues

- **SG_Vulgarity_Singlish (F1=0.42)**: Difficult to distinguish from Harassment
- **Cybersecurity_Malware (F1=0.36)**: Insufficient training data
- **SG_Racial_Religious_Harmony (F1=0.50)**: Needs more diverse examples
