# 二分类共享模型

## 对比

| 模型 | 平均 Macro F1 | 适配器大小 | 报告 |
|------|---------------|-----------|------|
| Multi-LoRA | 0.796 | 2MB/国 | [Multi_LoRA_shared.md](Multi_LoRA_shared.md) |
| Adapter | 0.769 | 2.5MB/国 | [Adapter_shared.md](Adapter_shared.md) |

## 各国违规召回率对比

| 国家 | Multi-LoRA Viol R | Adapter Viol R | 最佳 |
|------|-------------------|----------------|------|
| SA | 0.9832 | 0.4808 | **MLoRA** |
| BR | 0.9552 | 0.5224 | **MLoRA** |
| MX | 0.9489 | 0.5705 | **MLoRA** |
| ID | 0.9570 | 0.6335 | **MLoRA** |
| SG | 0.9260 | 0.7017 | **MLoRA** |
| TH | 0.9497 | 0.6141 | **MLoRA** |
| TR | 0.9266 | 0.5451 | **MLoRA** |
| ZA | 0.9604 | 0.3806 | **MLoRA** |
