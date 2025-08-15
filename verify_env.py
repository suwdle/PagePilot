import numpy as np
import config
from rl_env import PagePilotEnv
import os

def verify_environment():
    """Initializes the environment, performs one step, and prints detailed results."""
    print("--- Starting Environment Verification ---")

    env = None
    try:
        # 1. Initialize Environment
        env = PagePilotEnv(file_path=config.WEBSITE_PATH)
        print("Environment initialized.")

        # 2. Reset Environment and get initial state
        initial_obs = env.reset()
        print(f"\nInitial Observation (State): {initial_obs}")
        initial_potential = env.previous_potential
        print(f"Initial Potential: {initial_potential:.4f}")

        # 3. Define a single, deterministic action
        # Action map: {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', ...}
        action_to_take = 3  # Move Right
        action_name = 'Move Right'
        print(f"\n>>> Performing a single deterministic action: {action_name} (ID: {action_to_take})")

        # 4. Take the step
        new_obs, reward, done, info = env.step(action_to_take)

        # 5. Print the results
        print(f"\n--- Results after one step ---")
        print(f"New Observation (State): {new_obs}")
        
        # Recalculate potential for verification
        new_potential = env._get_potential(new_obs)
        print(f"New Potential: {new_potential:.4f}")
        
        print(f"Reward received (New Potential - Initial Potential): {reward:.4f}")
        print(f"Is Done? {done}")
        print(f"Info: {info}")

        # 6. Verify boundary conditions explicitly
        x, y, w, h, _ = new_obs
        is_out_of_bounds = (x < 0 or y < 0 or (x + w) > 1 or (y + h) > 1)
        print(f"Is element out of bounds? {is_out_of_bounds}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if env:
            env.close()
            print("\nEnvironment closed.")
        print("--- Environment Verification Finished ---")

if __name__ == "__main__":
    verify_environment()
