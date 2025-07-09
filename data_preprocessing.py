import pandas as pd
import os
from datasets import load_dataset

def analyze_wave_ui_data(ds):
    """
    Performs a deeper analysis of the WaveUI-25K dataset using pandas.
    """
    print("\n" + "#" * 20 + " WAVEUI DEEP ANALYSIS " + "#" * 20 + "\n")

    # Convert to pandas DataFrame
    df = ds['train'].to_pandas()

    # --- Categorical Feature Distribution ---
    print("--- Platform Distribution ---")
    print(df['platform'].value_counts())
    print("\n" + "="*50 + "\n")

    print("--- UI Element Type Distribution (Top 15) ---")
    print(df['type'].value_counts().nlargest(15))
    print("\n" + "="*50 + "\n")

    print("--- Language Distribution ---")
    print(df['language'].value_counts())
    print("\n" + "="*50 + "\n")

    # --- Numerical Feature Analysis ---
    # Resolution
    print("--- Screen Resolution Distribution (Top 10) ---")
    # Convert list to a hashable type (tuple) for value_counts
    resolutions = df['resolution'].apply(tuple)
    print(resolutions.value_counts().nlargest(10))
    print("\n" + "="*50 + "\n")

    # Bbox size
    print("--- Bounding Box Size Analysis ---")
    df['bbox_width'] = df['bbox'].apply(lambda x: x[2] - x[0])
    df['bbox_height'] = df['bbox'].apply(lambda x: x[3] - x[1])
    print(df[['bbox_width', 'bbox_height']].describe())
    print("\n" + "="*50 + "\n")

    print("WaveUI deep analysis data generated successfully!")

def load_and_preprocess_retailrocket_data(data_path):
    """
    Loads and preprocesses the RetailRocket dataset.
    """
    print("\n" + "#" * 20 + " RETAILROCKET PREPROCESSING " + "#" * 20 + "\n")

    # Define file paths
    events_path = os.path.join(data_path, 'events.csv')
    item_properties_part1_path = os.path.join(data_path, 'item_properties_part1.csv')
    item_properties_part2_path = os.path.join(data_path, 'item_properties_part2.csv')
    category_tree_path = os.path.join(data_path, 'category_tree.csv')

    # --- Load Data ---
    print("--- Loading RetailRocket Data ---")
    try:
        events_df = pd.read_csv(events_path)
        item_properties_df1 = pd.read_csv(item_properties_part1_path)
        item_properties_df2 = pd.read_csv(item_properties_part2_path)
        category_tree_df = pd.read_csv(category_tree_path)
        print("All RetailRocket CSV files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading RetailRocket file: {e}. Please ensure all files are in the specified path.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during RetailRocket data loading: {e}")
        return None, None, None

    # Combine item_properties
    item_properties_df = pd.concat([item_properties_df1, item_properties_df2], ignore_index=True)
    print(f"Combined item_properties_part1.csv and item_properties_part2.csv. Total rows: {len(item_properties_df)}")

    # Convert timestamps to datetime
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
    item_properties_df['timestamp'] = pd.to_datetime(item_properties_df['timestamp'], unit='ms')
    print("Timestamps converted to datetime objects.")

    print("RetailRocket data preprocessing complete.")
    return events_df, item_properties_df, category_tree_df


def main():
    """
    Loads the WaveUI-25K and RetailRocket datasets and performs analysis/preprocessing.
    """
    try:
        # --- WaveUI-25K Data ---
        print("\n" + "="*20 + " Processing WaveUI-25K Dataset " + "="*20 + "\n")
        ds_wave_ui = load_dataset("agentsea/wave-ui-25k")
        analyze_wave_ui_data(ds_wave_ui)

        # --- RetailRocket Data ---
        print("\n" + "="*20 + " Processing RetailRocket Dataset " + "="*20 + "\n")
        retailrocket_data_path = "/home/seokjun/pj/PagePilot/data/retailrocket/"
        events_df, item_properties_df, category_tree_df = load_and_preprocess_retailrocket_data(retailrocket_data_path)

        if events_df is not None:
            print("\nRetailRocket DataFrames loaded and preprocessed:")
            print(f"Events DataFrame shape: {events_df.shape}")
            print(f"Item Properties DataFrame shape: {item_properties_df.shape}")
            print(f"Category Tree DataFrame shape: {category_tree_df.shape}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
