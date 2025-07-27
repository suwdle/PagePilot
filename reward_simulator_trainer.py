
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import lightgbm as lgb
import numpy as np


def create_aggregated_features(raw_df):
    """
    Creates a pandas DataFrame with aggregated features per screenshot from raw element data.
    """
    df = raw_df.copy()

    # Basic feature extraction for each element
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['area'] = df['width'] * df['height']
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    df['text_density'] = df['ocr_text'].str.len() / (df['area'] + 1e-6)
    df['text_density'] = df['text_density'].fillna(0)

    # One-hot encode the 'type' column
    df_type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, df_type_dummies], axis=1)

    # --- Aggregate features per screenshot_id ---
    agg_dict = {
        'width': 'mean',
        'height': 'mean',
        'area': ['mean', 'sum'],
        'center_x': 'mean',
        'center_y': 'mean',
        'text_density': 'mean',
        **{col: 'sum' for col in df_type_dummies.columns}  # Count of each element type
    }

    # Group by screenshot_id and aggregate
    agg_df = df.groupby('screenshot_id').agg(agg_dict).reset_index()

    # Flatten MultiIndex columns
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns.values]
    agg_df = agg_df.rename(columns={'screenshot_id_': 'screenshot_id'})

    # Add a component count feature
    component_counts = df.groupby('screenshot_id').size().reset_index(name='component_count')
    agg_df = pd.merge(agg_df, component_counts, on='screenshot_id')

    return agg_df


def calculate_simulated_ctr(row, screen_width=1280, screen_height=720):
    """
    Calculates a simulated Click-Through Rate (CTR) for an entire UI page
    based on aggregated features.
    """
    ctr_score = 0.0

    # Heuristic 1: Bonus for having clickable elements
    clickable_elements_count = row.get('type_button_sum', 0) + row.get('type_link_sum', 0)
    ctr_score += clickable_elements_count * 0.1

    # Heuristic 2: Bonus for larger average element size
    avg_area = row.get('area_mean', 0)
    normalized_avg_area = avg_area / (screen_width * screen_height)
    ctr_score += normalized_avg_area * 5.0

    # Heuristic 3: Center proximity of the average element position
    avg_center_x, avg_center_y = row.get('center_x_mean', 0), row.get('center_y_mean', 0)
    screen_center_x, screen_center_y = screen_width / 2, screen_height / 2
    distance_from_center = np.sqrt((avg_center_x - screen_center_x)**2 + (avg_center_y - screen_center_y)**2)
    max_distance = np.sqrt(screen_center_x**2 + screen_center_y**2)
    proximity_score = 1 - (distance_from_center / max_distance)
    ctr_score += proximity_score * 0.5

    # Heuristic 4: Penalty for too many elements (clutter)
    component_count = row.get('component_count', 1)
    ctr_score -= (component_count / 50) * 0.2

    # Sigmoid function to keep the CTR between 0 and 1
    simulated_ctr = 1 / (1 + np.exp(- (ctr_score * 5 - 2)))
    return simulated_ctr


def main():
    """
    Main function to train and save the reward simulator model.
    """
    # Load the raw element dataset
    input_path = "./data/raw_elements.csv"
    print(f"Loading raw element data from {input_path}...")
    raw_df = pd.read_csv(input_path)

    # Dynamically create aggregated features
    print("Creating aggregated features from raw data...")
    features_df = create_aggregated_features(raw_df)

    # Generate reward labels
    print("Generating simulated CTR labels...")
    features_df['simulated_ctr'] = features_df.apply(calculate_simulated_ctr, axis=1)

    # Define features (X) and target (y)
    features = features_df.drop(columns=['screenshot_id', 'simulated_ctr'])
    features = features.fillna(0)
    target = features_df['simulated_ctr']

    # Sanitize feature names for LightGBM
    features.columns = [col.replace(' ', '_').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('=', '').replace(',', '') for col in features.columns]
    
    # Handle duplicate column names by appending a suffix
    cols = pd.Series(features.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i, i_val in enumerate(cols[cols == dup].index.values)]
    features.columns = cols

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Train a LightGBM Regressor model
    print("Training the LightGBM Regressor model...")
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on the test set: {mse:.6f}")

    # Save the trained model
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "reward_simulator_lgbm.joblib")
    print(f"Saving the trained model to {model_path}...")
    joblib.dump(model, model_path)

    print("Reward simulator training complete.")


if __name__ == "__main__":
    main()
