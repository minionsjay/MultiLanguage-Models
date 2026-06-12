# Indonesia (ID) 9-class Multi-Class Models

## Overview

| Version | Model | Data | Seed F1 | Cal F1 | Method |
|---------|-------|------|---------|--------|--------|
| v1 | 8-class Full FT | v3 (38,753) | 0.692 | — | Full FT + CE |
| v1 | 8-class LoRA | v3 (38,753) | 0.690 | — | LoRA r=16 + CE |
| v2 | 9-class LSD | v3 (38,753) | 0.674 | 0.684 | LoRA LSD T=3.0 |
| v5 | 2-seed LS Ensemble | v5 (42,253) | 0.695 | 0.709 | LoRA LS + ensemble + cal |
| **v6** | **LS single seed** | **v6 (45,753)** | **0.713** | **0.732** | **LoRA LS + cal** |

## Data Evolution

| Version | Rows | ID_SARA | Dangerous | Hate_Speech | Total Labels |
|---------|------|---------|-----------|-------------|-------------|
| v3 | 38,753 | 2,983 | 4,544 | 3,774 | 9 |
| v5 | 42,253 | 5,483 | 4,544 | 4,274 | 9 |
| v6 | 45,753 | 5,483 | 6,544 | 5,774 | 9 |

## Best Model: ID v6 (calibrated)

**Macro F1: 0.732 | Accuracy: 0.713**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|------|
| Political                      | 1.0000 | 0.8552 | 0.9219 |
| ID_Blasphemy                   | 0.9737 | 0.8810 | 0.9250 |
| safe                           | 0.7873 | 0.7926 | 0.7900 |
| Sexually_Explicit              | 0.7507 | 0.7783 | 0.7643 |
| Hate_Speech                    | 0.8097 | 0.5993 | 0.6888 |
| Dangerous_Content              | 0.7140 | 0.7475 | 0.7303 |
| ID_SARA                        | 0.6741 | 0.6241 | 0.6481 |
| Harassment                     | 0.4962 | 0.6182 | 0.5505 |
| other_violation                | 0.6615 | 0.5059 | 0.5733 |

### Training Parameters (v6)

```python
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
                         target_modules=["query_proj", "value_proj"])
learning_rate = 2e-4
epochs = 3
batch_size = 16 (train), 64 (eval)
label_smoothing = 0.05
weight_decay = 0.01
warmup_steps = 100
seed = 42
```

### Improvements

- ID_SARA: 0.42 → 0.62 → **0.65** (NIM data generation)
- Dangerous_Content: 0.58 → **0.73** (v6 supplement)
- Hate_Speech: 0.60 → **0.69** (v6 supplement)
