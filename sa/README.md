# Saudi Arabia (SA) Multi-Class Models

> 🇸🇦 Saudi content moderation | العربية | 8-9 labels

## Model Overview

| Model | Classes | Macro F1 | Acc | Method | Data |
|-------|---------|----------|-----|--------|------|
| **8-class Full FT** | 8 | **0.829** | **0.808** | Full FT, 3 epochs, lr=2e-5 | ~12,500 |
| Self-Distill + Cal | 9 | 0.796 | — | Cascade distill + threshold | ~15,000 |
| Distill + LS | 9 | 0.782 | 0.768 | Full FT distill + ls=0.05 | ~15,000 |
| Distill | 9 | 0.767 | 0.737 | Full FT distill T=3.0 | ~15,000 |
| Baseline 9-class | 9 | 0.609 | 0.566 | LoRA CE | ~15,000 |

## Labels

| # | Label | Description |
|---|-------|-------------|
| 1 | Dangerous_Content | Terrorism, violence, weapons |
| 2 | Harassment | Personal insults (سُبَّة, شتيمة) |
| 3 | Hate_Speech | Identity-based attacks |
| 4 | Political | Politically sensitive |
| 5 | SA_LGBTQ_Content | LGBTQ content (illegal in SA) |
| 6 | SA_Religious_Violation | Blasphemy, apostasy, religious insults |
| 7 | SA_State_Security_Royalty | Anti-government, anti-royal family |
| 8 | other_violation | Other violations |
| 9 | none (safe) | Normal content |

## Best Model: 8-Class Full FT

**Macro F1: 0.829 | Accuracy: 0.808**

### Per-Class

| Label | Precision | Recall | F1 |
|-------|-----------|--------|------|

## Training Parameters (8-Class Full FT)

```python
base_model = "microsoft/mdeberta-v3-base"
learning_rate = 2e-5
epochs = 3
batch_size = 16 (train), 32 (eval)
weight_decay = 0.01
warmup_steps = 100
# Full FT — no PEFT/LoRA
```

## Training Parameters (9-Class Distillation)

```python
# Teacher: Cascade Binary + 8-class Full FT
# Student: Full FT 9-class
T = 3.0, alpha = 0.5
learning_rate = 5e-5, epochs = 8
nim_gen_weight = 0.4
loss = alpha * CE + (1-alpha) * KL(teacher/T, student/T) * sample_weight
```

## Usage (8-Class Full FT)

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/sa_multiclass/model_8class_fullft_20260610_113559/final_model"
)
```

## Key Insight

SA's lower baseline (0.609) vs ID (0.692) was due to label quality. The 8-class Full FT (0.829) shows that excluding "none/safe" from multi-class significantly helps.
