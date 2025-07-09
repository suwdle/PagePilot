import pandas as pd
import os

def analyze_retailrocket_data(data_path):
    print("Starting RetailRocket data analysis...")

    # Define file paths
    events_path = os.path.join(data_path, 'events.csv')
    item_properties_part1_path = os.path.join(data_path, 'item_properties_part1.csv')
    item_properties_part2_path = os.path.join(data_path, 'item_properties_part2.csv')
    category_tree_path = os.path.join(data_path, 'category_tree.csv')

    # --- Load Data ---
    print("\n--- Loading Data ---")
    try:
        events_df = pd.read_csv(events_path)
        item_properties_df1 = pd.read_csv(item_properties_part1_path)
        item_properties_df2 = pd.read_csv(item_properties_part2_path)
        category_tree_df = pd.read_csv(category_tree_path)
        print("All CSV files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure all files are in the specified path.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # Combine item_properties
    item_properties_df = pd.concat([item_properties_df1, item_properties_df2], ignore_index=True)
    print(f"Combined item_properties_part1.csv and item_properties_part2.csv. Total rows: {len(item_properties_df)}")

    # --- Events Data Analysis ---
    print("\n--- Events Data Analysis (events.csv) ---")
    print(f"Shape: {events_df.shape}")
    print("\nInfo:")
    events_df.info()
    print("\nFirst 5 rows:")
    print(events_df.head())

    # Convert timestamp to datetime
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
    print("\nEvent types distribution:")
    print(events_df['event'].value_counts())

    print("\nUnique visitors and items:")
    print(f"Unique visitors: {events_df['visitorid'].nunique()}")
    print(f"Unique items: {events_df['itemid'].nunique()}")

    print("\nEvents over time (daily):")
    events_df['date'] = events_df['timestamp'].dt.date
    daily_events = events_df['date'].value_counts().sort_index()
    print(daily_events.head())
    print(daily_events.tail())

    # --- Item Properties Data Analysis ---
    print("\n--- Item Properties Data Analysis (item_properties.csv) ---")
    print(f"Shape: {item_properties_df.shape}")
    print("\nInfo:")
    item_properties_df.info()
    print("\nFirst 5 rows:")
    print(item_properties_df.head())

    print("\nUnique items with properties:")
    print(f"Unique items in properties: {item_properties_df['itemid'].nunique()}")

    print("\nTop 10 most frequent properties:")
    print(item_properties_df['property'].value_counts().head(10))

    # --- Category Tree Data Analysis ---
    print("\n--- Category Tree Data Analysis (category_tree.csv) ---")
    print(f"Shape: {category_tree_df.shape}")
    print("\nInfo:")
    category_tree_df.info()
    print("\nFirst 5 rows:")
    print(category_tree_df.head())

    print("\nNumber of unique categories:")
    print(f"Unique categories: {category_tree_df['categoryid'].nunique()}")

    print("\nCategories with parent categories (top 10):")
    print(category_tree_df.dropna(subset=['parentid']).head(10))

    print("\nRetailRocket data analysis complete.")

if __name__ == "__main__":
    # Assuming the data is in /home/seokjun/pj/PagePilot/data/retailrocket/
    retailrocket_data_path = "/home/seokjun/pj/PagePilot/data/retailrocket/"
    analyze_retailrocket_data(retailrocket_data_path)
