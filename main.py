import yaml
import os
from rl_env import LivePagePilotEnv
from dqn_trainer import train_dqn_live

def main():
    """
    Main entry point for the PagePilot application.
    Loads configuration, initializes the live environment, and starts the training/optimization process.
    """
    print("--- Starting PagePilot Optimization ---")

    # Load configuration
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return
    
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create and initialize the live environment
    print("Initializing live browser environment...")
    env = LivePagePilotEnv(config_path=config_path)

    # Start the DQN training process which performs the optimization
    print("Starting DQN agent for optimization...")
    train_dqn_live(env, config)

    # Close the environment
    print("Optimization finished. Closing environment.")
    env.close()

if __name__ == "__main__":
    main()