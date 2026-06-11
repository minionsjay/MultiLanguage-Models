# MultiLanguage Content Moderation Models

8 国 × 4 种训练方法 = 32 个二分类模型，基于 microsoft/mdeberta-v3-base。

## 覆盖国家

| 代码 | 国家 | 语种 |
|------|------|------|
| SA | Saudi Arabia | 阿拉伯语 |
| BR | Brazil | 葡萄牙语 |
| MX | Mexico | 西班牙语 |
| ID | Indonesia | 印尼语 |
| SG | Singapore | 英语/Singlish |
| TH | Thailand | 泰语 |
| TR | Turkey | 土耳其语 |
| ZA | South Africa | 英语/阿非利卡语 |

## 四种训练方法

| 方法 | 精度 | 模型大小 | 适用场景 |
|------|------|---------|---------|
| **Full FT** | 最高 (Avg F1: 0.853) | 1.1GB/国 | 生产环境 |
| **LoRA** | 较高 (Avg F1: 0.809) | 3MB adapter/国 | 快速部署 |
| **Multi-LoRA** | 中等 (Avg F1: 0.796) | 2MB adapter/国 | 多国共享基座 |
| **Adapter** | 一般 (Avg F1: 0.769) | 2.5MB/国 | 快速原型 |

## 模型报告

详见 [binary/README.md](binary/README.md) — 包含所有 32 个模型的完整评估指标（Precision/Recall/F1）。

```
binary/
├── fullft/       # Full Fine-Tuning (8 国)
├── lora/         # LoRA PEFT (8 国)
├── multilora/    # Multi-LoRA 共享 (8 国)
├── adapter/      # Bottleneck Adapter (8 国)
├── shared/       # 共享架构对比
├── usage_guide.py           # 使用代码
├── eval_binary_models.py    # 评估脚本
└── eval_results/            # 评估结果
```

## 使用

```python
# Full FT
from usage_guide import FullFTModel
model = FullFTModel("SG")
label, conf = model.predict("Wah lao this CB driver damn bodoh sia!")

# Multi-LoRA (8 国共享基座)
from usage_guide import MultiLoRAModel
model = MultiLoRAModel()
model.set_country("TH")
label, conf = model.predict("ไอ้เหี้ย เมิงจะเอาอะไร")
model.set_country("SG")
label, conf = model.predict("Wah lao...")
```

详见 [binary/README_USAGE.md](binary/README_USAGE.md)

## 模型权重

模型权重文件较大（Full FT: 1.1GB/国），存储在 HuggingFace Hub。

🤖 Generated with [Claude Code](https://claude.com/claude-code)
