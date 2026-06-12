# Saudi Arabia (SA) Multi-Class Models

## Overview

| Model | Classes | Macro F1 | Method |
|-------|---------|----------|--------|
| 8-class Full FT | 8 | 0.829 | Full FT |
| Self-Distill + Cal | 9 | 0.796 | Cascade distill + threshold |
| Distill + LS | 9 | 0.782 | Full FT distill + label smoothing |
| Distill | 9 | 0.767 | Full FT distill |
| Baseline 9-class | 9 | 0.609 | LoRA CE |

## Best Model: 8-class Full FT

**Macro F1: 0.829 | Accuracy: 0.808**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|------|

### Training Parameters

```python
learning_rate = 2e-5
epochs = 3
batch_size = 16
# Full FT — no PEFT
```
