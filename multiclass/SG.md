# Singapore (SG) Multi-Class Model

## Overview

| Attribute | Value |
|-----------|-------|
| Country | Singapore (SG) |
| Labels | 9 |
| Training Rows | 25,543 |
| Seed Macro F1 | 0.6686 |
| **Calibrated Macro F1** | **0.7043** |

## Per-Class Results

| Label | Precision | Recall | F1 |
|-------|-----------|--------|------|
| Politically_Sensitive_Topics        | 0.9897 | 0.9970 | 0.9934 |
| Hate_Speech                         | 0.9533 | 0.9985 | 0.9754 |
| safe                                | 0.8646 | 0.8278 | 0.8458 |
| weighted avg                        | 0.8204 | 0.8155 | 0.8168 |
| Dangerous_Content                   | 0.7716 | 0.7908 | 0.7811 |
| Sexually_Explicit_Information       | 0.7626 | 0.7076 | 0.7341 |
| Harassment                          | 0.7594 | 0.6827 | 0.7190 |
| macro avg                           | 0.7082 | 0.7048 | 0.7043 |
| SG_Racial_Religious_Harmony         | 0.5163 | 0.4922 | 0.5040 |
| SG_Vulgarity_Singlish               | 0.3626 | 0.5130 | 0.4249 |
| Cybersecurity_Malware               | 0.3939 | 0.3333 | 0.3611 |

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

| Cybersecurity_Malware               | 0.2418 |
| Dangerous_Content                   | 0.9367 |
| Harassment                          | 0.9473 |
| Hate_Speech                         | 0.7076 |
| Politically_Sensitive_Topics        | 0.8693 |
| SG_Racial_Religious_Harmony         | 0.4416 |
| SG_Vulgarity_Singlish               | 0.2524 |
| Sexually_Explicit_Information       | 0.5033 |
| safe                                | 0.5016 |
