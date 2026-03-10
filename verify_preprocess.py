import sys
import os
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.data.data_preprocessing import preprocess_text, preprocess_data

def test():
    logging.basicConfig(level=logging.INFO)
    
    # Test text cleaning
    sample_text = "I am LOVING this! 123 <br> It's better than expected."
    cleaned = preprocess_text(sample_text)
    print(f"Sample: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    # Create dummy raw data
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame({'Text': [sample_text, "Another review here.", "Bad! 1/10"]})
    df.to_csv("data/raw/train.csv", index=False)
    df.to_csv("data/raw/test.csv", index=False)
    
    # Run preprocessor
    processed_train = "data/processed/train.csv"
    preprocess_data("data/raw/train.csv", processed_train)
    
    if os.path.exists(processed_train):
        out_df = pd.read_csv(processed_train)
        print("Processed file content:")
        print(out_df[['Cleaned_Text']])
        print("Verification SUCCESSful")
    else:
        print("Verification FAILED - file not created")

if __name__ == "__main__":
    test()
