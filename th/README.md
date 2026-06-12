# Thailand (TH) Multi-Class Model

> 🇹🇭 Thai content moderation | ภาษาไทย | 8 labels

## Model Overview

| Attribute | Value |
|-----------|-------|
| Country | Thailand (TH) |
| Language | Thai |
| Task | 8-class content moderation |
| Method | LoRA Label Smoothing + Threshold Calibration |
| Base Model | microsoft/mdeberta-v3-base |
| Model Path | `Training/th_data/models/final_model` |
| Data Rows | 17,751 |
| Labels | 8 |

## Labels (8)

| # | Label | Description |
|---|-------|-------------|
| 1 | Dangerous_Content | Terrorism, violence, self-harm, drugs, fraud, CSAM |
| 2 | Harassment | Personal insults, bullying |
| 3 | Hate_Speech | Identity-based attacks |
| 4 | Sexually_Explicit_Information | Porn, arousal, hookup |
| 5 | Politically_Sensitive_Topics | NON-Thai politics |
| 6 | Cybersecurity_Malware | Malware, hacking, phishing |
| 7 | TH_Lese_Majeste | Anti-monarchy (Article 112 — lèse-majesté) |
| 8 | safe | Normal Thai life |

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
seed = 42
```

## Evaluation Results

**Seed Macro F1: 0.6768 | Calibrated Macro F1: 0.7440**

### Per-Class

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|--------|
| Hate_Speech                         | 0.9364 | 0.9822 | 0.9588 | 675 |
| Sexually_Explicit_Information       | 0.9343 | 0.8533 | 0.8920 | 150 |
| safe                                | 0.8629 | 0.8878 | 0.8751 | 900 |
| weighted avg                        | 0.8322 | 0.8333 | 0.8285 | 2663 |
| TH_Lese_Majeste                     | 0.7919 | 0.8138 | 0.8027 | 145 |
| Harassment                          | 0.8649 | 0.7218 | 0.7869 | 266 |
| macro avg                           | 0.7799 | 0.7268 | 0.7440 | 2663 |
| Dangerous_Content                   | 0.6301 | 0.7931 | 0.7023 | 290 |
| Cybersecurity_Malware               | 0.7544 | 0.4216 | 0.5409 | 102 |
| Politically_Sensitive_Topics        | 0.4646 | 0.3407 | 0.3932 | 135 |

### Thresholds

| Label | Threshold |
|-------|-----------|
| Cybersecurity_Malware               | 0.2328 |
| Dangerous_Content                   | 0.3765 |
| Harassment                          | 0.6224 |
| Hate_Speech                         | 0.2502 |
| Politically_Sensitive_Topics        | 0.2369 |
| Sexually_Explicit_Information       | 0.6179 |
| TH_Lese_Majeste                     | 0.5409 |
| safe                                | 0.3201 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/th_data/models/final_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/th_data/models/final_model"
)

labels = ["Dangerous_Content", "Harassment", "Hate_Speech",
          "Sexually_Explicit_Information", "Politically_Sensitive_Topics",
          "Cybersecurity_Malware", "TH_Lese_Majeste", "safe"]

def predict(text):
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, -1)
        pred = logits.argmax().item()
    return labels[pred], probs[0][pred].item()
```

## Data Source

- Primary: `data_v2/train_TH.csv` (11,253 rows, multi-class labels)
- Supplement: `Datasets_Nine_country/Thailand/` (4,787 safe samples)
- Generated: NIM API (Hate_Speech 1,500)

## Issues

- **Politically_Sensitive_Topics (F1=0.39)**: Confused with TH_Lese_Majeste
- **Cybersecurity_Malware (F1=0.54)**: Insufficient data
