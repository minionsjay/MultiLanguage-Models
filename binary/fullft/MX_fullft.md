# MX Full FT 二分类模型

## 模型概述

| 属性 | 值 |
|------|------|
| 国家 | Mexico (MX) |
| 语种 | 西班牙语 |
| 任务 | 二分类（safe/clean vs violation） |
| 训练方法 | Full Fine-Tuning（全量微调） |
| 基座模型 | microsoft/mdeberta-v3-base |
| 模型路径 | `Training/binary_8countries/MX_fullft/final_model` |
| 模型大小 | 1.14 GB |
| 数据源 | Datasets_Nine_country/Mexico/all_datasets_with_labels_dedup.csv |
| 训练脚本 | `Training/binary_8countries/train_all.py` |

## 训练参数

```python
model_name = "microsoft/mdeberta-v3-base"
learning_rate = 1e-5
num_train_epochs = 5  # 实际有 early stopping (patience=3)
per_device_train_batch_size = 4
per_device_eval_batch_size = 8
gradient_accumulation_steps = 4
max_grad_norm = 0.5
weight_decay = 0.01
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_length = 256
precision = FP32  # 全精度训练

# 损失函数: AdamW + class-weighted CrossEntropyLoss
# 训练时对 clean/violation 类别自动计算权重平衡
# 训练/验证分割: 85/15
```

## 评估结果

| 指标 | 值 |
|------|------|
| Accuracy | **0.8633** |
| Macro F1 | **0.8504** |
| Macro Precision | 0.8451 |
| Macro Recall | 0.8572 |

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|------|--------|
| clean (safe) | 0.9132 | 0.8763 | **0.8944** | 5789 |
| violation | 0.7770 | 0.8381 | **0.8064** | 2977 |

## 使用方法

### 方式一：直接加载

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/binary_8countries/MX_fullft/final_model",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/binary_8countries/MX_fullft/final_model",
    local_files_only=True
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax().item()

    labels = {0: "safe/clean", 1: "violation"}
    confidence = probs[0][pred].item()
    return labels[pred], confidence


# 示例
text = "some Mexico text to classify"
label, conf = predict(text)
print(f"Label: {label}, Confidence: {conf:.4f}")
```

### 方式二：Pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="/home/ninini/Agents/Colloation_data/Training/binary_8countries/MX_fullft/final_model",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

result = classifier("some text to classify")
print(result)
```

## 训练数据分布

- 总计: 8766 条
- clean (safe): 5789 条 (66.0%)
- violation: 2977 条 (34.0%)

## 限制

- **仅二分类**: 不能区分违规类型（hate speech vs dangerous content vs harassment 等）
- **单国家**: 该模型仅针对 Mexico 内容，不适用于其他国家
- **需要多分类时**: 请使用 ID v5/v6 或 SA 多分类模型

## 相关模型

- LoRA 版本: [MX_lora.md](MX_lora.md)（更小更快，精度略低）
- Multi-LoRA 共享: [shared/Multi_LoRA_shared.md](shared/Multi_LoRA_shared.md)（8 国共享基座）
