from typing import Dict, List

import pandas as pd


class RealEstateDataPreprocessor:
    """Prepare real estate data for LLM fine-tuning"""

    def prepare_training_data(self, market_data_df: pd.DataFrame, property_data: List[Dict]) -> List[Dict[str, str]]:
        training_examples = []
        for _, row in market_data_df.iterrows():
            region = row.get('REGION_NAME', 'Unknown')
            median_price = row.get('MEDIAN_SALE_PRICE', '')
            example = {
                "instruction": f"Summarize market for {region}",
                "input": f"What's the market like in {region}?",
                "output": f"Median price is {median_price}."
            }
            training_examples.append(example)
        return training_examples

    def detect_role_from_example(self, example: Dict[str, str]) -> str:
        text = example.get("instruction", "").lower() + example.get("input", "").lower()
        if "develop" in text:
            return "developer"
        if "buy" in text:
            return "buyer"
        if "agent" in text:
            return "agent"
        return "investor"
