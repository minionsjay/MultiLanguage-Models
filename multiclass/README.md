# Multi-Class Model Index

> All models use **microsoft/mdeberta-v3-base** | LoRA r=16 α=32 | Label Smoothing 0.05

## Models by Country

| Country | Labels | Rows | Method | Macro F1 | Report |
|---------|--------|------|--------|----------|--------|
| 🇸🇦 SA | 8 | ~12,500 | Full FT | **0.829** | [sa/](sa/) |
| 🇹🇭 TH | 8 | 17,751 | LoRA LS + cal | **0.744** | [th/](th/) |
| 🇮🇩 ID v6 | 9 | 45,753 | LoRA LS + cal | **0.732** | [id/](id/) |
| 🇸🇬 SG | 9 | 25,543 | LoRA LS + cal | **0.704** | [sg/](sg/) |
| 🇿🇦 ZA | 9 | 21,329 | LoRA LS + cal | **0.589** 🔄 | [za/](za/) |

## Directory Structure

```
multiclass/
├── README.md        ← this file
├── id/              ← Indonesia (6 versions, most developed)
├── sa/              ← Saudi Arabia (Full FT + distillation)
├── sg/              ← Singapore (new, 9 labels)
├── th/              ← Thailand (new, 8 labels)
└── za/              ← South Africa (supplementing)
```

## Common Training Recipe

```python
# All LoRA models use this config:
base_model = "microsoft/mdeberta-v3-base"
lora: r=16, alpha=32, dropout=0.1, target=["query_proj","value_proj"]
loss: CrossEntropyLoss(label_smoothing=0.05)
optimizer: AdamW, lr=2e-4, warmup=100 steps
epochs: 3, batch: 16/64
post: per-class grid search threshold + Nelder-Mead joint optimization
```

## Key Learnings

1. **NIM data generation is the most effective improvement** (ID_SARA +0.20, Dangerous +0.15)
2. **Full FT > LoRA for accuracy, LoRA > Full FT for deployment flexibility**
3. **Label Smoothing (0.05) consistently outperforms plain CE**
4. **Threshold calibration gives +0.01-0.02 stable gain across all models**
5. **LSD distillation is NOT worth the 5x training slowdown**
6. **Generated labels can overfit** (ZA Political 0.99) — need real data for country-specific labels
