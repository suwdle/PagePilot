
from datasets import load_dataset
import pandas as pd

def analyze_dataset(ds):
    """
    Analyzes the loaded Hugging Face dataset.
    """
    print("Dataset features:")
    print(ds['train'].features)
    print("\nSample data:")
    sample = ds['train'][0]
    print(sample)

def create_feature_dataframe(ds):
    """
    Creates a pandas DataFrame with extracted features from the dataset.
    """
    # Convert the dataset to a pandas DataFrame
    df = ds['train'].to_pandas()

    # Extract bounding box features
    df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['area'] = df['width'] * df['height']
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2

    # Extract text features
    df['text_density'] = df['OCR'].str.len() / (df['area'] + 1e-6)
    df['text_density'] = df['text_density'].fillna(0)

    # One-hot encode the 'type' column
    df_type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, df_type_dummies], axis=1)

    # Define the feature columns to keep
    feature_columns = [
        'width', 'height', 'area', 'center_x', 'center_y', 'text_density'
    ] + list(df_type_dummies.columns)

    return df[feature_columns]

def main():
    """
    Main function to download, analyze, and preprocess the dataset.
    """
    print("Downloading the dataset...")
    ds = load_dataset("agentsea/wave-ui-25k")
    print("Dataset downloaded successfully.")

    analyze_dataset(ds)

    print("\nExtracting features and creating DataFrame...")
    features_df = create_feature_dataframe(ds)

    output_path = "./data/preprocessed_waveui.csv"
    print(f"Saving preprocessed data to {output_path}...")
    features_df.to_csv(output_path, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
