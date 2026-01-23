import json
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path


class ClaimExtractor:
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["claim_extraction"]

        self.device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)

        self.max_new_tokens = cfg["max_new_tokens"]
        self.temperature = cfg["temperature"]

        prompt_path = Path(
            "experiments/claim_extraction/prompts/extract_claims.txt"
        )
        self.prompt_template = prompt_path.read_text()

    def extract(self, question: str, answer: str):
        prompt = self.prompt_template.format(
            question=question.strip(),
            answer=answer.strip()
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        try:
            claims = json.loads(decoded)
            if not isinstance(claims, list):
                return []
            return claims
        except json.JSONDecodeError:
            return []
