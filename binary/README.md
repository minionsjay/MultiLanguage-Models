# 二分类模型索引

> 基座: microsoft/mdeberta-v3-base | 8 国 × 4 种方法 = 32 个模型

## 目录结构

```
binary/
├── README.md              ← 本文件
├── usage_guide.py         ← 使用代码（4 种方法 × 8 国）
├── README_USAGE.md        ← 使用指南 + HuggingFace 上传说明
├── eval_binary_models.py  ← 批量评估脚本
├── eval_results/          ← 评估结果
├── fullft/                ← Full Fine-Tuning (8 国, 1.1GB/国)
├── lora/                  ← LoRA PEFT (8 国, 3MB/国)
├── multilora/             ← Multi-LoRA 共享 (8 国, 2MB/国)
├── adapter/               ← Bottleneck Adapter (8 国, 2.5MB/国)
└── shared/                ← 共享架构对比
```

---

## Full FT 模型 — [fullft/](fullft/)

> 最高精度，1.1GB/国，独立部署

| 国家 | Acc | Macro F1 | Clean F1 | Viol P | Viol R | Viol F1 | 报告 |
|------|-----|----------|----------|--------|--------|---------|------|
| SA | 0.8771 | 0.8736 | 0.8524 | 0.9001 | 0.8896 | 0.8948 | [report](fullft/SA_fullft.md) |
| BR | 0.8652 | 0.8288 | 0.9077 | 0.6876 | 0.8247 | 0.7500 | [report](fullft/BR_fullft.md) |
| MX | 0.8633 | 0.8504 | 0.8944 | 0.7770 | 0.8381 | 0.8064 | [report](fullft/MX_fullft.md) |
| ID | 0.8568 | 0.8568 | 0.8562 | 0.8488 | 0.8660 | 0.8573 | [report](fullft/ID_fullft.md) |
| SG | 0.9185 | 0.8807 | 0.9479 | 0.8060 | 0.8211 | 0.8135 | [report](fullft/SG_fullft.md) |
| TH | 0.8747 | 0.8675 | 0.8366 | 0.9154 | 0.8821 | 0.8984 | [report](fullft/TH_fullft.md) |
| TR | 0.8621 | 0.8270 | 0.7491 | 0.9086 | 0.9013 | 0.9049 | [report](fullft/TR_fullft.md) |
| ZA | 0.8491 | 0.8412 | 0.8058 | 0.8890 | 0.8646 | 0.8766 | [report](fullft/ZA_fullft.md) |

## LoRA 模型 — [lora/](lora/)

> PEFT LoRA r=16, 3MB/国

| 国家 | Acc | Macro F1 | Clean F1 | Viol P | Viol R | Viol F1 | 报告 |
|------|-----|----------|----------|--------|--------|---------|------|
| SA | 0.8358 | 0.8288 | 0.7943 | 0.8443 | 0.8833 | 0.8634 | [report](lora/SA_lora.md) |
| BR | 0.8290 | 0.7945 | 0.8787 | 0.6075 | 0.8550 | 0.7103 | [report](lora/BR_lora.md) |
| MX | 0.7846 | 0.7712 | 0.8266 | 0.6486 | 0.7985 | 0.7157 | [report](lora/MX_lora.md) |
| ID | 0.8283 | 0.8281 | 0.8214 | 0.8002 | 0.8723 | 0.8347 | [report](lora/ID_lora.md) |
| SG | 0.8856 | 0.8420 | 0.9250 | 0.6974 | 0.8324 | 0.7589 | [report](lora/SG_lora.md) |
| TH | 0.8396 | 0.8276 | 0.7821 | 0.8676 | 0.8787 | 0.8731 | [report](lora/TH_lora.md) |
| TR | 0.8190 | 0.7838 | 0.6966 | 0.9051 | 0.8393 | 0.8710 | [report](lora/TR_lora.md) |
| ZA | 0.7992 | 0.7914 | 0.7512 | 0.8658 | 0.8002 | 0.8317 | [report](lora/ZA_lora.md) |

## Multi-LoRA 模型 — [multilora/](multilora/)

> 共享基座 + 8 国 LoRA r=8, 2MB/国, 可热切换

| 国家 | Acc | Macro F1 | Clean F1 | Viol P | Viol R | Viol F1 | 报告 |
|------|-----|----------|----------|--------|--------|---------|------|
| SA | 0.9059 | 0.7950 | 0.6442 | 0.9111 | 0.9832 | 0.9458 | [report](multilora/SA_multilora.md) |
| BR | 0.8709 | 0.8031 | 0.6875 | 0.8848 | 0.9552 | 0.9187 | [report](multilora/BR_multilora.md) |
| MX | 0.8670 | 0.7947 | 0.6729 | 0.8862 | 0.9489 | 0.9165 | [report](multilora/MX_multilora.md) |
| ID | 0.8782 | 0.8170 | 0.7113 | 0.8909 | 0.9570 | 0.9228 | [report](multilora/ID_multilora.md) |
| SG | 0.8690 | 0.8100 | 0.7042 | 0.9058 | 0.9260 | 0.9158 | [report](multilora/SG_multilora.md) |
| TH | 0.8854 | 0.8494 | 0.7757 | 0.8979 | 0.9497 | 0.9231 | [report](multilora/TH_multilora.md) |
| TR | 0.8501 | 0.7804 | 0.6566 | 0.8827 | 0.9266 | 0.9041 | [report](multilora/TR_multilora.md) |
| ZA | 0.8504 | 0.7162 | 0.5210 | 0.8671 | 0.9604 | 0.9114 | [report](multilora/ZA_multilora.md) |

## Adapter 模型 — [adapter/](adapter/)

> Bottleneck Adapter, 2.5MB/国, 基座冻结

| 国家 | Acc | Macro F1 | Clean F1 | Viol P | Viol R | Viol F1 | 报告 |
|------|-----|----------|----------|--------|--------|---------|------|
| SA | 0.8786 | 0.7481 | 0.9294 | 0.6906 | 0.4808 | 0.5669 | [report](adapter/SA_adapter.md) |
| BR | 0.8474 | 0.7616 | 0.9046 | 0.7581 | 0.5224 | 0.6186 | [report](adapter/BR_adapter.md) |
| MX | 0.8511 | 0.7724 | 0.9062 | 0.7250 | 0.5705 | 0.6385 | [report](adapter/MX_adapter.md) |
| ID | 0.8804 | 0.8206 | 0.9242 | 0.8259 | 0.6335 | 0.7170 | [report](adapter/ID_adapter.md) |
| SG | 0.8541 | 0.7968 | 0.9047 | 0.6765 | 0.7017 | 0.6889 | [report](adapter/SG_adapter.md) |
| TH | 0.8659 | 0.8144 | 0.9122 | 0.8604 | 0.6141 | 0.7167 | [report](adapter/TH_adapter.md) |
| TR | 0.8311 | 0.7488 | 0.8926 | 0.6797 | 0.5451 | 0.6050 | [report](adapter/TR_adapter.md) |
| ZA | 0.8243 | 0.6790 | 0.8949 | 0.5914 | 0.3806 | 0.4632 | [report](adapter/ZA_adapter.md) |

---

## 违规召回率 TOP 10

| 排名 | 国家 | 方法 | Viol Recall | Viol F1 |
|------|------|------|-------------|---------|
| 1 | SA | MLoRA | 0.9832 | 0.9458 |
| 2 | ZA | MLoRA | 0.9604 | 0.9114 |
| 3 | ID | MLoRA | 0.9570 | 0.9228 |
| 4 | BR | MLoRA | 0.9552 | 0.9187 |
| 5 | TH | MLoRA | 0.9497 | 0.9231 |
| 6 | MX | MLoRA | 0.9489 | 0.9165 |
| 7 | TR | MLoRA | 0.9266 | 0.9041 |
| 8 | SG | MLoRA | 0.9260 | 0.9158 |
| 9 | TR | Full FT | 0.9013 | 0.9049 |
| 10 | SA | Full FT | 0.8896 | 0.8948 |

## 四种方法平均对比

| 方法 | 平均 Macro F1 | 每国大小 | 共享基座 | 推荐场景 |
|------|--------------|----------|---------|---------|
| **Full FT** | **0.853** | 1.1GB | ❌ | 生产环境，精度优先 |
| LoRA | 0.809 | 3MB | ❌ | 快速部署，接近 FT |
| Multi-LoRA | 0.796 | 2MB | ✅ 1.1GB | 多国服务，省空间 |
| Adapter | 0.769 | 2.5MB | ❌ | 快速原型 |
