# Adapter 共享二分类模型

## 架构

与 Multi-LoRA 类似，使用 bottleneck adapter：

```
mdeberta-v3-base (1.1GB, 冻结)
    ├── Adapter_SA (~2.5MB) → Saudi Arabia
    ├── Adapter_BR (~2.5MB) → Brazil
    ├── ... (8 国)
    └── Adapter_ZA (~2.5MB) → South Africa
```

## 使用方法

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/adapter/SG/final_model",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/adapter/SG/final_model",
    local_files_only=True
)

def predict(text):
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    return "safe" if pred == 0 else "violation"
```

## 训练参数

```python
learning_rate = 1e-4, epochs = 3
batch_size = 32 (train), 64 (eval)
weight_decay = 0.01, warmup_ratio = 0.1
lr_scheduler = "cosine", max_length = 128, precision = FP32
# 基座冻结，仅训练 adapter bottleneck + 分类头
# Adapter 参数: 1,491,272 (0.5% of 279M)
data: data_v2/train_{COUNTRY}.csv, split 90/10
```

## 各国完整结果

### 整体指标

| 国家 | Accuracy | Macro F1 | Macro P | Macro R |
|------|----------|----------|---------|---------|
| SA | 0.8786 | 0.7481 | 0.7968 | 0.7191 |
| BR | 0.8474 | 0.7616 | 0.8115 | 0.7353 |
| MX | 0.8511 | 0.7724 | 0.8020 | 0.7528 |
| ID | 0.8804 | 0.8206 | 0.8593 | 0.7958 |
| SG | 0.8541 | 0.7968 | 0.7932 | 0.8007 |
| TH | 0.8659 | 0.8144 | 0.8638 | 0.7881 |
| TR | 0.8311 | 0.7488 | 0.7732 | 0.7326 |
| ZA | 0.8243 | 0.6790 | 0.7249 | 0.6576 |

### 各类别 Precision / Recall / F1

| 国家 | Clean P | Clean R | Clean F1 | Viol P | Viol R | Viol F1 |
|------|---------|---------|----------|--------|--------|---------|
| SA | 0.9030 | 0.9573 | 0.9294 | 0.6906 | 0.4808 | 0.5669 |
| BR | 0.8648 | 0.9483 | 0.9046 | 0.7581 | 0.5224 | 0.6186 |
| MX | 0.8790 | 0.9352 | 0.9062 | 0.7250 | 0.5705 | 0.6385 |
| ID | 0.8926 | 0.9580 | 0.9242 | 0.8259 | 0.6335 | 0.7170 |
| SG | 0.9098 | 0.8997 | 0.9047 | 0.6765 | 0.7017 | 0.6889 |
| TH | 0.8673 | 0.9620 | 0.9122 | 0.8604 | 0.6141 | 0.7167 |
| TR | 0.8667 | 0.9201 | 0.8926 | 0.6797 | 0.5451 | 0.6050 |
| ZA | 0.8585 | 0.9346 | 0.8949 | 0.5914 | 0.3806 | 0.4632 |

### 违规召回率排名

| 排名 | 国家 | Viol Recall | Viol F1 |
|------|------|-------------|---------|
| 1 | SG | 0.7017 | 0.6889 |
| 2 | ID | 0.6335 | 0.7170 |
| 3 | TH | 0.6141 | 0.7167 |
| 4 | MX | 0.5705 | 0.6385 |
| 5 | TR | 0.5451 | 0.6050 |
| 6 | BR | 0.5224 | 0.6186 |
| 7 | SA | 0.4808 | 0.5669 |
| 8 | ZA | 0.3806 | 0.4632 |

## 与 Multi-LoRA 对比

| 维度 | Multi-LoRA | Adapter |
|------|-----------|---------|
| PEFT 方法 | LoRA (低秩分解) | Bottleneck Adapter |
| 平均 Macro F1 | 0.796 | 0.769 |
| 平均 Viol Recall | — | — |
| 可训参数 | 0.21% | 0.5% |
| 适配器大小 | ~2MB | ~2.5MB |
| 库依赖 | peft | adapters |
