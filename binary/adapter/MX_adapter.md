# MX Adapter 二分类模型

## 模型概述

| 属性 | 值 |
|------|------|
| 国家 | Mexico (MX) |
| 任务 | 二分类（safe/clean vs violation） |
| 训练方法 | Adapter（Bottleneck Adapter, 基座冻结） |
| 基座模型 | microsoft/mdeberta-v3-base（冻结） |
| 模型路径 | `Training/adapter/MX/adapter_weights` |
| 可训参数 | 1,491,272 (0.5% of 279M) |
| 适配器大小 | ~2.5MB |
| 数据源 | data_v2/train_MX.csv |
| 训练脚本 | `Training/adapter/train_adapter.py` |
| 库依赖 | adapter-transformers |

## 训练参数

```python
learning_rate = 1e-4
epochs = 3
batch_size = 32 (train), 64 (eval)
weight_decay = 0.01
warmup_ratio = 0.1
lr_scheduler = "cosine"
max_length = 128
precision = FP32
train/val split = 90/10

# Adapter 特有配置
# - 基座完全冻结
# - 在 Transformer 层之间插入 bottleneck adapter
# - 仅训练 adapter 参数 + 分类头
# - adapter 参数: 1,491,272 (0.5%)
```

## 评估结果

### 整体指标

| 指标 | 值 |
|------|------|
| Accuracy | **0.8511** |
| Macro F1 | **0.7724** |
| Macro Precision | 0.8020 |
| Macro Recall | 0.7528 |

### 各类别详细

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|------|--------|
| **clean (safe)** | 0.8790 | 0.9352 | **0.9062** | 1018 |
| **violation** | 0.7250 | 0.5705 | **0.6385** | 305 |

## 与 Full FT / Multi-LoRA 对比

| 指标 | Adapter | Full FT | Multi-LoRA | 最佳 |
|------|---------|---------|------------|------|
| Accuracy | 0.8511 | 0.8633 | 0.8511 | **Full FT** |
| Macro F1 | 0.7724 | 0.8504 | 0.7724 | **Full FT** |
| Clean F1 | 0.9062 | 0.8944 | 0.9062 | **Full FT** |
| Viol F1 | 0.6385 | 0.8064 | 0.6385 | **Full FT** |
| Viol Recall | 0.5705 | 0.8381 | 0.5705 | — |
| Clean Recall | 0.9352 | 0.8763 | 0.9352 | — |

## 使用方法

```python
# Adapter 模型使用 adapter-transformers 库
# pip install adapter-transformers

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 直接加载已保存的完整模型（含 adapter）
model = AutoModelForSequenceClassification.from_pretrained(
    "/path/to/adapter/MX/adapter_weights"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/adapter/MX/adapter_weights"
)

def predict(text):
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax().item()
    return "violation" if pred == 1 else "safe"
```

## 特点

- **训练最快**: 仅训练 0.5% 参数
- **适配器最小**: ~2.5MB，比 LoRA (r=16, ~3MB) 略大
- **精度最低**: 在所有方法中整体 F1 最低
- **Clean 召回率高**: Adapter 偏向判定为 safe
- **Violation 召回率低**: 容易漏判违规内容
- **适用场景**: 快速原型验证，不建议生产使用
