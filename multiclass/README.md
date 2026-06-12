# Multi-Class Model Index

| Country | Labels | Rows | Method | Macro F1 | Report |
|---------|--------|------|--------|----------|--------|
| 🇸🇦 SA 8cl | 8 | ~12,500 | Full FT | **0.829** | [SA.md](SA.md) |
| 🇹🇭 TH | 8 | 17,751 | LoRA LS + cal | **0.744** | [TH.md](TH.md) |
| 🇮🇩 ID v6 | 9 | 45,753 | LoRA LS + cal | **0.732** | [ID.md](ID.md) |
| 🇸🇬 SG | 9 | 25,543 | LoRA LS + cal | **0.704** | [SG.md](SG.md) |
| 🇿🇦 ZA | 9 | 21,329 | LoRA LS + cal | **0.589** | [ZA.md](ZA.md) |

## Common Config

```python
Base: microsoft/mdeberta-v3-base
LoRA: r=16, alpha=32, dropout=0.1
Loss: CrossEntropyLoss(label_smoothing=0.05)
lr=2e-4, epochs=3, batch=16/64
Threshold cal: per-class grid + Nelder-Mead joint opt
```

## Next Steps

- 🇿🇦 ZA: Generate ZA_Xenophobia + ZA_Severe_Racism supplement data (like ID_SARA)
- 🇲🇽 MX: Prepare data
- 🇧🇷 BR: Prepare data
- 🇹🇷 TR: Prepare data
