# Binary Model Usage Guide

## 文件说明

| 文件 | 说明 |
|------|------|
| `usage_guide.py` | 完整使用代码（4 种方法 × 8 国） |
| `eval_binary_models.py` | 批量评估脚本 |
| `eval_results/` | 评估结果（JSON/CSV/Markdown） |

## 快速开始

### 1. Full FT 模型

```python
from usage_guide import FullFTModel

model = FullFTModel("SG")  # Singapore Full FT model
label, confidence = model.predict("Wah lao this CB driver damn bodoh sia!")
# -> ("violation", 0.9876)

# 批量推理
results = model.predict_batch([
    "Shiok lah the laksa damn power!",
    "KNN you better watch out I find you I whack you!"
])
```

### 2. LoRA 模型

```python
from usage_guide import LoRAModel

model = LoRAModel("SG")  # PEFT LoRA, ~2MB adapter
label, confidence = model.predict("Some Singlish text here...")
```

### 3. Multi-LoRA（8 国共享基座）

```python
from usage_guide import MultiLoRAModel

model = MultiLoRAModel()  # 一个基座服务 8 国

# 新加坡
model.set_country("SG")
label, conf = model.predict("Wah lao this damn shiok sia!")

# 切换到泰国
model.set_country("TH")
label, conf = model.predict("ไอ้เหี้ย เมิงจะเอาอะไร")
```

### 4. Adapter 模型

```python
from usage_guide import AdapterModel

model = AdapterModel("SG")
label, confidence = model.predict("Some text...")
```

### 5. 统一 API

```python
from usage_guide import ContentModerationAPI

# 一行切换方法
api = ContentModerationAPI(method="fullft", country="SG")
result = api.classify("Wah lao...")
print(result)  # {"text": ..., "label": "violation", "confidence": 0.95}

# 批量
results = api.classify_batch(["text1", "text2", "text3"])
```

## CLI 使用

```bash
# 单条推理
python usage_guide.py --method fullft --country SG --text "Wah lao damn bodoh sia!"

# 批量推理
python usage_guide.py --method lora --country TH --file thai_texts.txt --output results.json

# Multi-LoRA 演示
python usage_guide.py --method multilora --country SG
```

## 上传到 HuggingFace

### Full FT 模型

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载本地模型
model = AutoModelForSequenceClassification.from_pretrained(
    "/path/to/binary_8countries/SG_fullft/final_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/binary_8countries/SG_fullft/final_model"
)

# 上传
model.push_to_hub("your-org/mdeberta-v3-SG-binary-fullft")
tokenizer.push_to_hub("your-org/mdeberta-v3-SG-binary-fullft")
```

### LoRA 模型

```python
from peft import PeftModel

# 加载 base + adapter
base_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/mdeberta-v3-base", num_labels=2
)
model = PeftModel.from_pretrained(
    base_model,
    "/path/to/binary_8countries/SG_lora/final_model"
)

# 上传 adapter（不是完整模型）
model.push_to_hub("your-org/mdeberta-v3-SG-binary-lora")
```

### Multi-LoRA 适配器

```python
# 只上传每个国家的 LoRA adapter（2MB/个）
# base_model 使用 microsoft/mdeberta-v3-base
model.push_to_hub("your-org/mdeberta-v3-multilora-SG")
```

## 依赖

```bash
pip install transformers torch peft scikit-learn
# Adapter 模型额外需要:
# pip install adapter-transformers
```
