from datasets import load_dataset
import pandas as pd
import json

def analyze_dataset(ds):
    """
    Analyzes the loaded Hugging Face dataset.
    """
    print("Dataset features:")
    print(ds['train'].features)
    print("\nSample data:")
    # Inspect the first sample
    sample = ds['train'][0]
    print(sample)

def extract_features(data):
    """
    Extracts statistical features from a single data point.
    """
    features = {}

    # UI element counts
    element_types = ['button', 'link', 'input', 'text', 'image']
    for element_type in element_types:
        features[f'{element_type}_count'] = 0

    # For simplicity, we'll use the 'type' field to count elements.
    # A more robust approach would involve deeper analysis of the data.
    if data['type'] in element_types:
        features[f'{data["type"]}_count'] = 1

    # Bounding box features
    if data['bbox']:
        x1, y1, x2, y2 = data['bbox']
        width = x2 - x1
        height = y2 - y1
        features['avg_width'] = width
        features['avg_height'] = height
        features['avg_area'] = width * height
        features['center_x'] = (x1 + x2) / 2
        features['center_y'] = (y1 + y2) / 2

    # Text features
    if data['OCR']:
        features['text_density'] = len(data['OCR']) / (features.get('avg_area', 1) + 1e-6)
    else:
        features['text_density'] = 0

    # UI complexity
    features['ui_complexity'] = 1 # In this simplified case, each entry is one element

    return features


def main():
    """
    Main function to download, analyze, and preprocess the dataset.
    """
    # Load the dataset
    print("Downloading the dataset...")
    ds = load_dataset("agentsea/wave-ui-25k")
    print("Dataset downloaded successfully.")

    # Analyze the dataset structure
    analyze_dataset(ds)

    # Extract features and create a DataFrame
    print("\nExtracting features...")
    feature_list = [extract_features(item) for item in ds['train']]
    df = pd.DataFrame(feature_list)

    # Save the preprocessed data
    output_path = "/home/seokjun/pj/PagePilot/data/preprocessed_waveui.csv"
    print(f"Saving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()