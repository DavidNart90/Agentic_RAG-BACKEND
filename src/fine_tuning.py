from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer


class RealEstateLLMFineTuner:
    """Simple fine-tuner wrapper for local models"""

    def __init__(self, base_model: str = "microsoft/DialoGPT-medium"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)

    def fine_tune(self, training_data: List[Dict[str, str]], output_dir: str):
        """Placeholder fine-tune method - saves base model to output_dir"""
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        return None
