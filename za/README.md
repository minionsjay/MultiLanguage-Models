# South Africa (ZA) Multi-Class Model

> 🇿🇦 South African content moderation | English/Afrikaans/Zulu/Xhosa | 9 labels

## Model Overview

| Attribute | Value |
|-----------|-------|
| Country | South Africa (ZA) |
| Languages | English, Afrikaans, isiZulu, isiXhosa, etc. (12 official) |
| Task | 9-class content moderation |
| Method | LoRA Label Smoothing + Threshold Calibration |
| Base Model | microsoft/mdeberta-v3-base |
| Model Path | `Training/za_data/models/final_model` |
| Data Rows | 21,329 |
| Labels | 9 |

## Labels (9)

| # | Label | Description |
|---|-------|-------------|
| 1 | Dangerous_Content | Terrorism, weapons, severe violence, drugs, fraud |
| 2 | Harassment | Personal insults (poes, doos, naai, fok) |
| 3 | Hate_Speech | Identity-based attacks |
| 4 | Sexually_Explicit_Information | Porn, arousal |
| 5 | Politically_Sensitive_Topics | NON-SA politics |
| 6 | Cybersecurity_Malware | Malware, hacking |
| 7 | ZA_Severe_Racism | K-word, apartheid slurs, racial violence |
| 8 | ZA_Xenophobia | Xenophobic attacks on African foreigners |
| 9 | safe | Normal SA life |

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

**Seed Macro F1: 0.5711 | Calibrated Macro F1: 0.5886**

### Per-Class

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|--------|
| Politically_Sensitive_Topics        | 0.9766 | 1.0000 | 0.9882 | 209 |
| Hate_Speech                         | 0.9322 | 0.9778 | 0.9544 | 225 |
| safe                                | 0.7000 | 0.8253 | 0.7575 | 1162 |
| Harassment                          | 0.7689 | 0.6868 | 0.7256 | 281 |
| Sexually_Explicit_Information       | 0.6811 | 0.7069 | 0.6937 | 290 |
| weighted avg                        | 0.6471 | 0.6672 | 0.6468 | 3200 |
| macro avg                           | 0.6175 | 0.5905 | 0.5886 | 3200 |
| Dangerous_Content                   | 0.5704 | 0.5448 | 0.5573 | 424 |
| Cybersecurity_Malware               | 0.2508 | 0.3571 | 0.2947 | 210 |
| ZA_Xenophobia                       | 0.4000 | 0.1400 | 0.2074 | 200 |
| ZA_Severe_Racism                    | 0.2778 | 0.0754 | 0.1186 | 199 |

### Thresholds

| Label | Threshold |
|-------|-----------|
| Cybersecurity_Malware               | 0.3665 |
| Dangerous_Content                   | 0.4544 |
| Harassment                          | 0.9424 |
| Hate_Speech                         | 0.9366 |
| Politically_Sensitive_Topics        | 0.7788 |
| Sexually_Explicit_Information       | 0.3297 |
| ZA_Severe_Racism                    | 0.5085 |
| ZA_Xenophobia                       | 0.4928 |
| safe                                | 0.6689 |

## Usage

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/za_data/models/final_model"
)
```

## Data Source

- Primary: `data_v2/train_ZA.csv` (14,510 rows, multi-class labels)
- Supplement: `Datasets_Nine_country/South-Africa/` (5,000 safe samples)
- Generated: NIM API (Hate_Speech + Politically_Sensitive_Topics)

## Issues & Next Steps

- **ZA_Xenophobia (F1=0.21)** and **ZA_Severe_Racism (F1=0.12)**: SEVERELY underperforming
- Generated labels (Political 0.99, Hate_Speech 0.95) are overfitting to NIM patterns
- 🔄 Currently generating +2,500 each for ZA_Xenophobia and ZA_Severe_Racism
- Will retrain after supplement data is ready
