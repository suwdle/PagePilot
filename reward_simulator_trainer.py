
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

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

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Train a baseline Linear Regression model
    print("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on the test set: {mse:.6f}")

    # Save the trained model
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "reward_simulator_lr.joblib")
    print(f"Saving the trained model to {model_path}...")
    joblib.dump(model, model_path)

    print("Reward simulator training complete.")

if __name__ == "__main__":
    main()
