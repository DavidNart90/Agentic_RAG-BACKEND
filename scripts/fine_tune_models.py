from src.fine_tuning import RealEstateLLMFineTuner
import json


def fine_tune_role_models():
    """Fine-tune separate models for each role"""
    roles = ['investor', 'developer', 'buyer', 'agent']
    for role in roles:
        print(f"Fine-tuning {role} model...")
        with open(f'training_data/{role}_training.jsonl', 'r') as f:
            training_data = [json.loads(line) for line in f]
        fine_tuner = RealEstateLLMFineTuner(base_model="microsoft/DialoGPT-medium")
        fine_tuner.fine_tune(training_data, output_dir=f"models/{role}_llm")
        print(f"✅ {role} model fine-tuned and saved")

if __name__ == "__main__":
    fine_tune_role_models()
