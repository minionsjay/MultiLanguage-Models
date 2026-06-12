# South Africa (ZA) Multi-Class Model

## Overview

| Attribute | Value |
|-----------|-------|
| Labels | 9 |
| Training Rows | 21,329 |
| Seed Macro F1 | 0.5711 |
| **Calibrated Macro F1** | **0.5886** |

## Per-Class Results

| Label | Precision | Recall | F1 |
|-------|-----------|--------|------|
| Politically_Sensitive_Topics        | 0.9766 | 1.0000 | 0.9882 |
| Hate_Speech                         | 0.9322 | 0.9778 | 0.9544 |
| safe                                | 0.7000 | 0.8253 | 0.7575 |
| Harassment                          | 0.7689 | 0.6868 | 0.7256 |
| Sexually_Explicit_Information       | 0.6811 | 0.7069 | 0.6937 |
| weighted avg                        | 0.6471 | 0.6672 | 0.6468 |
| macro avg                           | 0.6175 | 0.5905 | 0.5886 |
| Dangerous_Content                   | 0.5704 | 0.5448 | 0.5573 |
| Cybersecurity_Malware               | 0.2508 | 0.3571 | 0.2947 |
| ZA_Xenophobia                       | 0.4000 | 0.1400 | 0.2074 |
| ZA_Severe_Racism                    | 0.2778 | 0.0754 | 0.1186 |

## Issues

- **ZA_Xenophobia (F1=0.21)** and **ZA_Severe_Racism (F1=0.12)** are severely underperforming
- Generated labels (Political 0.99, Hate_Speech 0.95) are overfitting to NIM patterns
- Need more ZA-specific data for Xenophobia and Severe Racism labels

## Training Parameters

```python
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)
learning_rate = 2e-4, epochs = 3
label_smoothing = 0.05, seed = 42
```
