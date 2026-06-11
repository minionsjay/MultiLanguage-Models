# Multi-LoRA 共享二分类模型

## 架构

```
mdeberta-v3-base (1.1GB, 共享)
    ├── LoRA_SA (r=8, ~2MB) → Saudi Arabia
    ├── LoRA_BR (r=8, ~2MB) → Brazil
    ├── LoRA_MX (r=8, ~2MB) → Mexico
    ├── LoRA_ID (r=8, ~2MB) → Indonesia
    ├── LoRA_SG (r=8, ~2MB) → Singapore
    ├── LoRA_TH (r=8, ~2MB) → Thailand
    ├── LoRA_TR (r=8, ~2MB) → Turkey
    └── LoRA_ZA (r=8, ~2MB) → South Africa

总磁盘: 1.1GB (基座) + 8×2MB = ≈1.12GB
```

## 使用方法

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

# 加载基座
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/mdeberta-v3-base", num_labels=2
)

# 加载需要的国家适配器
model = PeftModel.from_pretrained(
    base_model,
    "/home/ninini/Agents/Colloation_data/Training/multi_lora/SG/final_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ninini/Agents/Colloation_data/Training/multi_lora/SG/final_model"
)

def predict(text, country='SG'):
    # 切换国家: 重新加载对应 LoRA
    # model.load_adapter(f"path/to/{country}/final_model", adapter_name=country)
    # model.set_adapter(country)
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax().item()
        prob = torch.softmax(logits, -1)[0][pred].item()
    return ("safe" if pred == 0 else "violation"), prob
```

## 训练参数

```python
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                         target_modules=["query_proj", "value_proj"])
learning_rate = 2e-4
epochs = 5 (early_stopping patience=3)
batch_size = 32 (train), 64 (eval)
weight_decay = 0.01, warmup_ratio = 0.1
lr_scheduler = "cosine", max_length = 128, precision = BF16
data: data_v2/train_{COUNTRY}.csv, split 90/10
```

## 各国完整结果

### 整体指标

| 国家 | Accuracy | Macro F1 | Macro P | Macro R |
|------|----------|----------|---------|---------|
| SA | 0.9059 | 0.7950 | 0.8850 | 0.7493 |
| BR | 0.8709 | 0.8031 | 0.8454 | 0.7773 |
| MX | 0.8670 | 0.7947 | 0.8315 | 0.7712 |
| ID | 0.8782 | 0.8170 | 0.8560 | 0.7922 |
| SG | 0.8690 | 0.8100 | 0.8192 | 0.8020 |
| TH | 0.8854 | 0.8494 | 0.8713 | 0.8334 |
| TR | 0.8501 | 0.7804 | 0.8009 | 0.7654 |
| ZA | 0.8504 | 0.7162 | 0.7933 | 0.6844 |

### 各类别 Precision / Recall / F1

| 国家 | Clean P | Clean R | Clean F1 | Viol P | Viol R | Viol F1 |
|------|---------|---------|----------|--------|--------|---------|
| SA | 0.8590 | 0.5154 | 0.6442 | 0.9111 | 0.9832 | 0.9458 |
| BR | 0.8060 | 0.5994 | 0.6875 | 0.8848 | 0.9552 | 0.9187 |
| MX | 0.7768 | 0.5934 | 0.6729 | 0.8862 | 0.9489 | 0.9165 |
| ID | 0.8211 | 0.6273 | 0.7113 | 0.8909 | 0.9570 | 0.9228 |
| SG | 0.7326 | 0.6780 | 0.7042 | 0.9058 | 0.9260 | 0.9158 |
| TH | 0.8447 | 0.7170 | 0.7757 | 0.8979 | 0.9497 | 0.9231 |
| TR | 0.7190 | 0.6042 | 0.6566 | 0.8827 | 0.9266 | 0.9041 |
| ZA | 0.7195 | 0.4083 | 0.5210 | 0.8671 | 0.9604 | 0.9114 |

### 违规召回率 (Violation Recall) 排名

| 排名 | 国家 | Viol Recall | Viol P | Viol F1 |
|------|------|-------------|--------|---------|
| 1 | SA | 0.9832 | 0.9111 | 0.9458 |
| 2 | ZA | 0.9604 | 0.8671 | 0.9114 |
| 3 | ID | 0.9570 | 0.8909 | 0.9228 |
| 4 | BR | 0.9552 | 0.8848 | 0.9187 |
| 5 | TH | 0.9497 | 0.8979 | 0.9231 |
| 6 | MX | 0.9489 | 0.8862 | 0.9165 |
| 7 | TR | 0.9266 | 0.8827 | 0.9041 |
| 8 | SG | 0.9260 | 0.9058 | 0.9158 |

### 安全内容召回率 (Clean Recall) 排名

| 排名 | 国家 | Clean Recall | Clean P | Clean F1 |
|------|------|-------------|---------|----------|
| 1 | TH | 0.7170 | 0.8447 | 0.7757 |
| 2 | SG | 0.6780 | 0.7326 | 0.7042 |
| 3 | ID | 0.6273 | 0.8211 | 0.7113 |
| 4 | TR | 0.6042 | 0.7190 | 0.6566 |
| 5 | BR | 0.5994 | 0.8060 | 0.6875 |
| 6 | MX | 0.5934 | 0.7768 | 0.6729 |
| 7 | SA | 0.5154 | 0.8590 | 0.6442 |
| 8 | ZA | 0.4083 | 0.7195 | 0.5210 |

## 特点

- **优点**: 一个基座服务 8 国，切换只加载 2MB LoRA 权重
- **优点**: 磁盘占用仅 1.12GB（8 国独立 Full FT 需 8×1.1GB = 8.8GB）
- **缺点**: 准确率低于独立 Full FT（平均 F1 0.796 vs 0.853）
- **注意**: 违规召回率高但 clean 召回率低，对安全内容误判较多
