# ZA Multi-LoRA 二分类模型

## 模型概述

| 属性 | 值 |
|------|------|
| 国家 | South Africa (ZA) |
| 任务 | 二分类（safe/clean vs violation） |
| 训练方法 | Multi-LoRA（共享基座 + 国家专属 LoRA） |
| 基座模型 | microsoft/mdeberta-v3-base |
| 适配器路径 | `Training/multi_lora/ZA/lora_weights` |
| 适配器大小 | ~2MB (r=8) |
| 数据源 | data_v2/train_ZA.csv |
| 训练脚本 | `Training/multi_lora/train_multi_lora.py` |

## 训练参数

```python
# 与独立 LoRA 的关键差异
lora_config = LoraConfig(
    r=8,                    # < 独立 LoRA 的 r=16
    lora_alpha=16,          # < 独立 LoRA 的 alpha=32
    lora_dropout=0.1,
    target_modules=["query_proj", "value_proj"]
)
learning_rate = 2e-4        # > 独立 LoRA 的 5e-5
epochs = 5 (early_stopping patience=3)
batch_size = 32 (train), 64 (eval)   # > 独立 LoRA 的 16/32
weight_decay = 0.01
warmup_ratio = 0.1
lr_scheduler = "cosine"
max_length = 128            # < 独立 LoRA 的 256
precision = BF16
train/val split = 90/10
# 特点: 8 国共享基座, 所有国家一起训练
```

## 评估结果

### 整体指标

| 指标 | 值 |
|------|------|
| Accuracy | **0.8504** |
| Macro F1 | **0.7162** |
| Macro Precision | 0.7933 |
| Macro Recall | 0.6844 |

### 各类别详细

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|------|--------|
| **clean (safe)** | 0.7195 | 0.4083 | **0.5210** | 289 |
| **violation** | 0.8671 | 0.9604 | **0.9114** | 1162 |

## 与 Full FT / LoRA 对比

| 指标 | Multi-LoRA | Full FT | Δ |
|------|-----------|---------|------|
| Accuracy | 0.8504 | 0.8491 | +0.0013 |
| Macro F1 | 0.7162 | 0.8412 | -0.1250 |
| Clean F1 | 0.5210 | 0.8058 | -0.2848 |
| Viol F1 | 0.9114 | 0.8766 | +0.0348 |
| **Viol Recall** | **0.9604** | **0.8646** | **+0.0958** |
| Clean Recall | 0.4083 | 0.8238 | -0.4155 |

## 使用方法

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载共享基座
base = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/mdeberta-v3-base", num_labels=2
)

# 加载此国家的 LoRA 适配器
model = PeftModel.from_pretrained(
    base,
    "/path/to/multi_lora/ZA/lora_weights"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/multi_lora/ZA/lora_weights"
)

# 推理
def predict(text):
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax().item()
    return "violation" if pred == 1 else "safe"
```

## 特点

- **优点**: 基座共享，切换国家只需加载 2MB LoRA
- **优点**: 违规召回率较高（Multi-LoRA 对违规信号更敏感）
- **缺点**: Clean 召回率低，容易将安全内容误判为违规
- **缺点**: 整体精度低于独立 Full FT 模型
