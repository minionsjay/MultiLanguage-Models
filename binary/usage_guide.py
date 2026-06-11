#!/usr/bin/env python3
"""
Binary Content Moderation Model Usage Guide
===========================================
Usage examples for all 4 training methods: Full FT, LoRA, Multi-LoRA, Adapter.

Supports 8 countries: SA, BR, MX, ID, SG, TH, TR, ZA
Model paths should be updated to match deployment location.

Usage:
    python usage_guide.py --method fullft --country SG --text "some text"
    python usage_guide.py --method lora --country TH --file texts.txt
"""

import argparse
import torch
import os
from typing import List, Tuple, Dict, Optional

# ============================================================
# Configuration — update these paths for your deployment
# ============================================================
# Option A: Local paths
LOCAL_PATHS = {
    "fullft": "/path/to/binary_8countries/{country}_fullft/final_model",
    "lora": "/path/to/binary_8countries/{country}_lora/final_model",
    "multilora_base": "/path/to/mdeberta-v3-base",
    "multilora_adapter": "/path/to/multi_lora/{country}/lora_weights",
    "adapter": "/path/to/adapter/{country}/adapter_weights",
}

# Option B: HuggingFace Hub
HF_PATHS = {
    "fullft": "your-org/mdeberta-v3-{country}-binary-fullft",
    "lora": "your-org/mdeberta-v3-{country}-binary-lora",
    "multilora_base": "microsoft/mdeberta-v3-base",
    "multilora_adapter": "your-org/mdeberta-v3-multilora-{country}",
    "adapter": "your-org/mdeberta-v3-{country}-binary-adapter",
}

COUNTRIES = {
    "SA": "Saudi Arabia", "BR": "Brazil", "MX": "Mexico",
    "ID": "Indonesia", "SG": "Singapore", "TH": "Thailand",
    "TR": "Turkey", "ZA": "South Africa",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Full Fine-Tuning Model
# ============================================================
class FullFTModel:
    """
    Standard HuggingFace model — load with AutoModelForSequenceClassification.

    Use when: you want maximum accuracy, don't mind 1.1GB per country.

    Example:
        model = FullFTModel("SG", use_hf=False, local_base="/models")
        label, conf = model.predict("Wah lao this driver damn bodoh sia")
    """

    def __init__(self, country: str, use_hf: bool = False, local_base: str = ""):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.country = country
        if use_hf:
            path = HF_PATHS["fullft"].format(country=country)
        else:
            path = LOCAL_PATHS["fullft"].format(country=country)
            if local_base:
                path = os.path.join(local_base, f"{country}_fullft/final_model")

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict single text. Returns (label, confidence)."""
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()

        return ("violation" if pred == 1 else "safe", probs[0][pred].item())

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Batch predict. Returns list of {'text': ..., 'label': ..., 'confidence': ..., 'prob_safe': ..., 'prob_violation': ...}"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=256, return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            for j, text in enumerate(batch):
                results.append({
                    "text": text,
                    "label": "violation" if preds[j].item() == 1 else "safe",
                    "confidence": probs[j][preds[j]].item(),
                    "prob_safe": probs[j][0].item(),
                    "prob_violation": probs[j][1].item(),
                })
        return results


# ============================================================
# 2. LoRA Model (PEFT)
# ============================================================
class LoRAModel:
    """
    PEFT LoRA model — base model + LoRA adapter (~2MB).

    Use when: you want smaller deployment size, slightly lower accuracy than Full FT.

    Example:
        model = LoRAModel("SG", use_hf=False, local_base="/models")
        label, conf = model.predict("Wah lao this driver damn bodoh sia")
    """

    def __init__(self, country: str, use_hf: bool = False, local_base: str = ""):
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.country = country

        # Load base model
        base_path = HF_PATHS["multilora_base"] if use_hf else LOCAL_PATHS["multilora_base"]
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=2, ignore_mismatched_sizes=True
        )

        # Load LoRA adapter
        if use_hf:
            adapter_path = HF_PATHS["lora"].format(country=country)
        else:
            adapter_path = LOCAL_PATHS["lora"].format(country=country)
            if local_base:
                adapter_path = os.path.join(local_base, f"{country}_lora/final_model")

        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()

        return ("violation" if pred == 1 else "safe", probs[0][pred].item())

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=256, return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            for j, text in enumerate(batch):
                results.append({
                    "text": text,
                    "label": "violation" if preds[j].item() == 1 else "safe",
                    "confidence": probs[j][preds[j]].item(),
                    "prob_safe": probs[j][0].item(),
                    "prob_violation": probs[j][1].item(),
                })
        return results


# ============================================================
# 3. Multi-LoRA Model (shared base + country adapters)
# ============================================================
class MultiLoRAModel:
    """
    Shared base model + 8 country LoRA adapters (2MB each).
    Hot-switch between countries with load_adapter().

    Use when: you need to serve multiple countries with one base model.

    Example:
        model = MultiLoRAModel(use_hf=False, local_base="/models")

        # Classify SG text
        model.set_country("SG")
        label, conf = model.predict("Wah lao this driver damn bodoh sia")

        # Switch to TH
        model.set_country("TH")
        label, conf = model.predict("ไอ้เหี้ย นี่มันอะไรกันวะ")
    """

    def __init__(self, use_hf: bool = False, local_base: str = ""):
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Load base model once
        base_path = HF_PATHS["multilora_base"] if use_hf else LOCAL_PATHS["multilora_base"]
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=2, ignore_mismatched_sizes=True
        )
        self.base_model.to(DEVICE)
        self.base_model.eval()

        self.use_hf = use_hf
        self.local_base = local_base
        self.current_country = None
        self.model = None
        self.tokenizer = None

        # Pre-load all tokenizers (small)
        self.tokenizers = {}
        for code in COUNTRIES:
            if use_hf:
                adapter_path = HF_PATHS["multilora_adapter"].format(country=code)
            else:
                adapter_path = LOCAL_PATHS["multilora_adapter"].format(country=code)
                if local_base:
                    adapter_path = os.path.join(local_base, f"multi_lora/{code}/lora_weights")
            try:
                self.tokenizers[code] = AutoTokenizer.from_pretrained(adapter_path)
            except:
                pass

    def set_country(self, country: str):
        """Switch to a different country's LoRA adapter."""
        from peft import PeftModel

        if country == self.current_country and self.model is not None:
            return

        if self.use_hf:
            adapter_path = HF_PATHS["multilora_adapter"].format(country=country)
        else:
            adapter_path = LOCAL_PATHS["multilora_adapter"].format(country=country)
            if self.local_base:
                adapter_path = os.path.join(self.local_base, f"multi_lora/{country}/lora_weights")

        # Load new PeftModel with country's LoRA
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.to(DEVICE)
        self.model.eval()
        self.current_country = country

        if country in self.tokenizers:
            self.tokenizer = self.tokenizers[country]

    def predict(self, text: str, country: str = None) -> Tuple[str, float]:
        if country:
            self.set_country(country)
        if self.model is None:
            raise RuntimeError("No country set. Call set_country() first.")

        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()

        return ("violation" if pred == 1 else "safe", probs[0][pred].item())

    def predict_batch(self, texts: List[str], country: str = None,
                      batch_size: int = 32) -> List[Dict]:
        if country:
            self.set_country(country)
        if self.model is None:
            raise RuntimeError("No country set.")

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=128, return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            for j, text in enumerate(batch):
                results.append({
                    "text": text,
                    "label": "violation" if preds[j].item() == 1 else "safe",
                    "confidence": probs[j][preds[j]].item(),
                    "prob_safe": probs[j][0].item(),
                    "prob_violation": probs[j][1].item(),
                })
        return results


# ============================================================
# 4. Adapter Model
# ============================================================
class AdapterModel:
    """
    Adapter-transformers model — bottleneck adapter (~2.5MB).

    Use when: using adapter-transformers library specifically.

    NOTE: Requires the 'adapters' library (pip install adapter-transformers).
    The saved adapter_weights directories need to be used with
    adapter-transformers loading mechanisms.

    Example:
        model = AdapterModel("SG")
        label, conf = model.predict("Wah lao this driver damn bodoh sia")
    """

    def __init__(self, country: str, use_hf: bool = False, local_base: str = ""):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.country = country

        if use_hf:
            path = HF_PATHS["adapter"].format(country=country)
        else:
            path = LOCAL_PATHS["adapter"].format(country=country)
            if local_base:
                path = os.path.join(local_base, f"adapter/{country}/adapter_weights")

        # Adapter models saved with full weights (includes adapter)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()

        return ("violation" if pred == 1 else "safe", probs[0][pred].item())


# ============================================================
# 5. Unified API
# ============================================================
class ContentModerationAPI:
    """
    Unified API for all 4 modeling approaches.

    Example:
        # Full FT
        api = ContentModerationAPI(method="fullft", country="SG")
        result = api.classify("Wah lao this driver damn bodoh sia")

        # Multi-LoRA with country switching
        api = ContentModerationAPI(method="multilora")
        api.set_country("TH")
        result = api.classify("ไอ้เหี้ย")
        api.set_country("SG")
        result = api.classify("Wah lao...")
    """

    def __init__(self, method: str = "fullft", country: str = None,
                 use_hf: bool = False, local_base: str = ""):
        self.method = method
        self.country = country

        if method == "fullft":
            self.model = FullFTModel(country, use_hf, local_base)
        elif method == "lora":
            self.model = LoRAModel(country, use_hf, local_base)
        elif method == "multilora":
            self.model = MultiLoRAModel(use_hf, local_base)
        elif method == "adapter":
            self.model = AdapterModel(country, use_hf, local_base)
        else:
            raise ValueError(f"Unknown method: {method}")

    def classify(self, text: str, country: str = None) -> Dict:
        """Classify a single text."""
        if self.method == "multilora":
            label, conf = self.model.predict(text, country)
        else:
            label, conf = self.model.predict(text)

        return {"text": text, "label": label, "confidence": conf}

    def classify_batch(self, texts: List[str], country: str = None) -> List[Dict]:
        """Classify multiple texts."""
        if self.method == "multilora":
            return self.model.predict_batch(texts, country)
        else:
            return self.model.predict_batch(texts)

    def set_country(self, country: str):
        """Switch country (Multi-LoRA only)."""
        if self.method == "multilora":
            self.model.set_country(country)
        else:
            raise ValueError("set_country() only available for Multi-LoRA")


# ============================================================
# CLI Interface
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Binary content moderation model inference"
    )
    parser.add_argument("--method", default="fullft",
                        choices=["fullft", "lora", "multilora", "adapter"])
    parser.add_argument("--country", default="SG",
                        choices=list(COUNTRIES.keys()))
    parser.add_argument("--text", help="Single text to classify")
    parser.add_argument("--file", help="File with texts (one per line)")
    parser.add_argument("--use-hf", action="store_true",
                        help="Use HuggingFace Hub paths")
    parser.add_argument("--local-base", default="",
                        help="Base directory for local models")
    parser.add_argument("--output", help="Output JSON file for batch results")

    args = parser.parse_args()

    print(f"Loading model: {args.method} | country: {args.country}")
    print(f"Device: {DEVICE}")

    api = ContentModerationAPI(
        method=args.method, country=args.country,
        use_hf=args.use_hf, local_base=args.local_base
    )

    if args.text:
        result = api.classify(args.text)
        print(f"\nText: {args.text[:100]}...")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")

    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Classifying {len(texts)} texts...")
        results = api.classify_batch(texts)

        viol_count = sum(1 for r in results if r['label'] == 'violation')
        print(f"Violation: {viol_count}/{len(texts)} ({viol_count/len(texts)*100:.1f}%)")
        print(f"Safe: {len(texts)-viol_count}/{len(texts)}")

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")

    else:
        # Demo mode
        print("\nDemo mode — testing with sample texts:\n")
        demos = {
            "SG": [
                "Wah lao this CB driver almost bang my car, damn bodoh sia!",
                "Shiok lah the chicken rice at Maxwell damn power, confirm must try!",
            ],
            "TH": [
                "ไอ้เหี้ย เมิงจะเอาอะไรนักหนาวะ กูไม่สนแล้ว",
                "วันนี้ไปกินข้าวที่โรงอาหาร อร่อยมากเลยครับ",
            ],
            "ID": [
                "Anjing lu! Mampus aja sana, gak punya otak!",
                "Makan siang enak banget di warteg tadi, kenyang pol.",
            ],
        }
        samples = demos.get(args.country, [["test text 1", "test text 2"]])
        for text in samples:
            result = api.classify(text)
            emoji = "🚨" if result['label'] == 'violation' else "✅"
            print(f"  {emoji} [{result['label']}] {text[:80]}...")
            print(f"     confidence: {result['confidence']:.4f}\n")


if __name__ == "__main__":
    main()
