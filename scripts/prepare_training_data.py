import pandas as pd
import json
from src.data_preparation import RealEstateDataPreprocessor

def create_training_dataset():
    """Convert existing data into LLM training format"""
    market_data = pd.read_csv('data/market_analytics.csv')
    with open('data/property_listings.json', 'r') as f:
        property_data = json.load(f)
    preprocessor = RealEstateDataPreprocessor()
    training_examples = preprocessor.prepare_training_data(market_data, property_data)

    role_datasets = {
        'investor': [],
        'developer': [],
        'buyer': [],
        'agent': []
    }

    for example in training_examples:
        role = preprocessor.detect_role_from_example(example)
        role_datasets[role].append(example)

    for role, examples in role_datasets.items():
        with open(f'training_data/{role}_training.jsonl', 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

    print(f"Generated {len(training_examples)} training examples")
    return role_datasets

if __name__ == "__main__":
    create_training_dataset()
