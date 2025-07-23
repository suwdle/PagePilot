
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import lightgbm as lgb

def main():
    """
    Main function to train and save the reward simulator model.
    """
    # Load the labeled dataset
    input_path = "./data/labeled_waveui.csv"
    print(f"Loading labeled data from {input_path}...")
    df = pd.read_csv(input_path)

    # Define features (X) and target (y)
    # Drop non-feature columns and fill NaNs just in case
    features = df.drop(columns=['simulated_ctr'])
    features = features.fillna(0)
    target = df['simulated_ctr']

    # Sanitize feature names for LightGBM
    features.columns = [col.replace(' ', '_').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace('=', '').replace(',', '') for col in features.columns]

    # Handle duplicate column names by appending a suffix
    cols = pd.Series(features.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i, i_val in enumerate(cols[cols == dup].index.values)]
    features.columns = cols

    print("Sanitized and unique feature names:", features.columns.tolist())

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
