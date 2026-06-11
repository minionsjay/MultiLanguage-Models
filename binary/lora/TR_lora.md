# TR LoRA 二分类模型

## 模型概述

| 属性 | 值 |
|------|------|
| 国家 | Turkey (TR) |
| 语种 | 土耳其语 |
| 任务 | 二分类（safe/clean vs violation） |
| 训练方法 | LoRA (PEFT, Low-Rank Adaptation) |
| 基座模型 | microsoft/mdeberta-v3-base |
| 模型路径 | `Training/binary_8countries/TR_lora/final_model` |
| 模型大小 | 0.02 GB |
| 可训参数 | 596,745 (0.21% of 279M) |
| 数据源 | Datasets_Nine_country/Turkey/all_datasets_with_labels_dedup.csv |
| 训练脚本 | `Training/binary_8countries/train_all.py` |

## 训练参数

```python
model_name = "microsoft/mdeberta-v3-base"

# LoRA 配置
lora_config = LoraConfig(
    r=16,                    # 低秩维度
    lora_alpha=32,           # 缩放因子
    lora_dropout=0.1,        # dropout
    target_modules=["query_proj", "value_proj"]  # 仅训练 attention 投影层
)

# 训练超参数
learning_rate = 5e-5
num_train_epochs = 5  # 实际有 early stopping (patience=3)
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
weight_decay = 0.01
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_length = 256
precision = BF16  # 半精度加速

# 训练/验证分割: 85/15
```

## 评估结果

### 整体指标

| 指标 | 值 |
|------|------|
| Accuracy | **0.8190** |
| Macro F1 | **0.7838** |
| Macro Precision | 0.7725 |
| Macro Recall | 0.8019 |

### 各类别详细指标

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|------|--------|
| **clean (safe)** | 0.6398 | 0.7644 | **0.6966** | 416 |
| **violation** | 0.9051 | 0.8393 | **0.8710** | 1114 |

### 混淆矩阵解读

```
                    预测 clean    预测 violation
实际 clean           TP_clean        FN (漏判)
实际 violation       FP (误判)       TP_violation
```

- **clean recall = 0.7644**: 76.4% 的安全内容被正确识别
- **violation recall = 0.8393**: 83.9% 的违规内容被正确召回
- **clean precision = 0.6398**: 模型判为 clean 的内容中 64.0% 确实是 clean
- **violation precision = 0.9051**: 模型判为 violation 的内容中 90.5% 确实是违规

## 使用方法

### 方式一：PeftModel 加载（推荐）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

# 1. 加载基座（需要指定 num_labels=2）
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/mdeberta-v3-base",
    num_labels=2,
    local_files_only=True  # 如果离线
)

# 2. 加载 LoRA 适配器
model = PeftModel.from_pretrained(
    base_model,
    "/home/ninini/Agents/Colloation_data/Training/binary_8countries/TR_lora/final_model"
)
model = model.merge_and_unload()  # 合并 LoRA 到基座，方便推理

tokenizer = AutoTokenizer.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/binary_8countries/TR_lora/final_model",
    local_files_only=True
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict(text):
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=256, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax().item()

    labels = {0: "safe/clean", 1: "violation"}
    confidence = probs[0][pred].item()
    return labels[pred], confidence


# 示例
text = "some Turkey social media post..."
label, conf = predict(text)
print(f"Label: {label}, Confidence: {conf:.4f}")
```

### 方式二：已合并模型直接加载

```python
# 如果模型已经 merge_and_unload 保存
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/binary_8countries/TR_lora/final_model",
    local_files_only=True
)
```

## 与 Full FT 对比

| 指标 | Full FT | LoRA | Δ |
|------|---------|------|------|
| Accuracy | 0.8621 | 0.8190 | -0.0431 |
| Macro F1 | 0.8270 | 0.7838 | -0.0432 |
| Clean F1 | 0.7491 | 0.6966 | -0.0525 |
| Viol F1 | 0.9049 | 0.8710 | -0.0339 |
| 训练参数 | 279M | 597K (0.21%) | - |
| 训练速度 | 基准 | 更快 | - |

## 限制

- **仅二分类**: 不能区分违规类型
- **单国家**: 仅针对 Turkey
- **精度低于 Full FT**: 适合快速部署，生产环境建议 Full FT

## 相关模型

- Full FT 版本: [TR_fullft.md](TR_fullft.md)（更高精度）
- Multi-LoRA 共享: [shared/Multi_LoRA_shared.md](shared/Multi_LoRA_shared.md)（8 国共享基座）
