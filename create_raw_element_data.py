
from datasets import load_dataset
import pandas as pd
import os

def create_raw_element_data():
    """
    Downloads the WaveUI-25K dataset and saves the raw element data to a CSV file
    without any aggregation. This will serve as the single source of truth.
    """
    print("Downloading the dataset...")
    # Using streaming=True to handle large datasets efficiently if needed
    ds = load_dataset("agentsea/wave-ui-25k", split='train', streaming=False)
    print("Dataset downloaded successfully.")

    # Convert to pandas DataFrame
    df = pd.DataFrame(ds)

    # --- Feature Selection and Renaming ---
    # Select only the necessary columns for the raw data
    # We can expand this later if more features are needed
    required_columns = {
        'screenshot_id': 'screenshot_id',
        'type': 'type',
        'bbox': 'bbox',
        'OCR': 'ocr_text',
        'purpose': 'purpose'
    }
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing from the dataset: {missing_cols}")

    df = df[list(required_columns.keys())]
    df = df.rename(columns=required_columns)

    # --- Data Cleaning and Formatting ---
    # Expand the bounding box list into separate columns for easier access
    df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
    df = df.drop(columns=['bbox'])

    # Fill potential NaN values in text fields
    df['ocr_text'] = df['ocr_text'].fillna('')
    df['purpose'] = df['purpose'].fillna('')

    # Ensure correct data types
    df = df.astype({
        'screenshot_id': 'str',
        'type': 'category',
        'x1': 'float32',
        'y1': 'float32',
        'x2': 'float32',
        'y2': 'float32',
        'ocr_text': 'str',
        'purpose': 'str'
    })

    # Create the data directory if it doesn't exist
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "raw_elements.csv")
    print(f"Saving raw element data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} elements.")

if __name__ == "__main__":
    create_raw_element_data()
