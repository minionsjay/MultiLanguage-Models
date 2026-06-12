# Thailand (TH) Multi-Class Model

## Overview

| Attribute | Value |
|-----------|-------|
| Country | Thailand (TH) |
| Labels | 8 |
| Training Rows | 17,751 |
| Seed Macro F1 | 0.6768 |
| **Calibrated Macro F1** | **0.7440** |

## Per-Class Results

| Label | Precision | Recall | F1 |
|-------|-----------|--------|------|
| Hate_Speech                         | 0.9364 | 0.9822 | 0.9588 |
| Sexually_Explicit_Information       | 0.9343 | 0.8533 | 0.8920 |
| safe                                | 0.8629 | 0.8878 | 0.8751 |
| weighted avg                        | 0.8322 | 0.8333 | 0.8285 |
| TH_Lese_Majeste                     | 0.7919 | 0.8138 | 0.8027 |
| Harassment                          | 0.8649 | 0.7218 | 0.7869 |
| macro avg                           | 0.7799 | 0.7268 | 0.7440 |
| Dangerous_Content                   | 0.6301 | 0.7931 | 0.7023 |
| Cybersecurity_Malware               | 0.7544 | 0.4216 | 0.5409 |
| Politically_Sensitive_Topics        | 0.4646 | 0.3407 | 0.3932 |

## Training Parameters

```python
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
                         target_modules=["query_proj", "value_proj"])
learning_rate = 2e-4
epochs = 3
batch_size = 16 (train), 64 (eval)
label_smoothing = 0.05
weight_decay = 0.01
seed = 42
```

## Thresholds

| Cybersecurity_Malware               | 0.2328 |
| Dangerous_Content                   | 0.3765 |
| Harassment                          | 0.6224 |
| Hate_Speech                         | 0.2502 |
| Politically_Sensitive_Topics        | 0.2369 |
| Sexually_Explicit_Information       | 0.6179 |
| TH_Lese_Majeste                     | 0.5409 |
| safe                                | 0.3201 |
