import torch
import pandas as pd
import numpy as np
from rl_env import PagePilotEnv
from dqn_trainer import DQN

def evaluate_agent(env, policy_net, test_df):
    """
    Evaluates the DQN agent's performance on a test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_rewards = []
    optimized_rewards = []

    # Select only the feature columns for evaluation and ensure numeric type
    feature_df = test_df[env.state_columns].astype(np.float32)

    for index, initial_state_series in feature_df.iterrows():
        # 1. Calculate initial reward
        initial_reward = env.reward_model.predict(initial_state_series.values.reshape(1, -1))[0]
        initial_rewards.append(initial_reward)

        # 2. Optimize UI with the agent
        state = torch.tensor(initial_state_series.values, dtype=torch.float32, device=device).unsqueeze(0)

        print(f"--- Optimizing UI for sample {index} ---")
        print(f"Initial State (first 5 features): {initial_state_series.values[:5]}")
        print(f"Initial Predicted CTR: {initial_reward:.4f}")

        for step in range(10): # Let the agent take 10 steps to optimize
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)

            new_state_series = env._take_action(action.item(), pd.Series(state.squeeze(0).cpu().numpy(), index=env.state_columns))
            state = torch.tensor(new_state_series.values, dtype=torch.float32, device=device).unsqueeze(0)

            optimized_reward = env._get_reward(new_state_series)
            print(f"  Step {step+1}: Action={action.item()}, New CTR={optimized_reward:.4f}")

        # 3. Calculate final optimized reward
        final_reward = env._get_reward(new_state_series)
        optimized_rewards.append(final_reward)
        print(f"Final Predicted CTR: {final_reward:.4f}\n")

    # 4. Calculate and print average performance
    avg_initial_reward = np.mean(initial_rewards)
    avg_optimized_reward = np.mean(optimized_rewards)
    improvement = ((avg_optimized_reward - avg_initial_reward) / avg_initial_reward) * 100 if avg_initial_reward != 0 else float('inf')

    print("--- Evaluation Summary ---")
    print(f"Average Initial Predicted CTR: {avg_initial_reward:.4f}")
    print(f"Average Optimized Predicted CTR: {avg_optimized_reward:.4f}")
    print(f"Improvement: {improvement:.2f}%")

def main():
    """
    Main function to load the model and run the evaluation.
    """
    # Load environment and data
    data_path = "./data/labeled_waveui.csv"
    reward_model_path = "./models/reward_simulator_lgbm.joblib"
    env = PagePilotEnv(data_path, reward_model_path)

    # Use a subset of the data for testing
    test_df = pd.read_csv(data_path).head(10) 
    
    # Load the trained DQN model
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN(input_dim, output_dim).to(device)
    policy_net.load_state_dict(torch.load("./models/dqn_model.pth", map_location=device))
    policy_net.eval()

    # Run the evaluation
    evaluate_agent(env, policy_net, test_df)

if __name__ == "__main__":
    main()
