

import gym
from gym import spaces
import numpy as np
import pandas as pd
import config
import joblib
import os

class PagePilotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, persona='casual_browser'):
        super(PagePilotEnv, self).__init__()
        
        # Load the trained LightGBM model
        model_path = 'models/reward_simulator_lgbm.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Reward simulator model not found at {model_path}. Please run train_reward_simulator.py first.")
        self.reward_simulator = joblib.load(model_path)
        self.persona = persona

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps_per_episode = config.MAX_STEPS_PER_EPISODE
        self.current_state = None
        self.previous_potential = 0

    def _get_potential(self, state):
        """Calculates the potential of a state using the loaded LGBM model."""
        x, y, width, height, _ = state
        
        # Create a feature DataFrame that matches the model's training data
        features = {
            'pos_x': x,
            'pos_y': y,
            'size': width * height,  # Use area as a proxy for size
            'contrast': 1.0,  # Using a neutral placeholder for contrast
            'persona_power_user': 1 if self.persona == 'power_user' else 0,
            'persona_casual_browser': 1 if self.persona == 'casual_browser' else 0,
            'persona_elderly': 1 if self.persona == 'elderly' else 0,
        }
        feature_df = pd.DataFrame([features])

        # Ensure column order is the same as during training
        training_columns = [
            'pos_x', 'pos_y', 'size', 'contrast',
            'persona_power_user', 'persona_casual_browser', 'persona_elderly'
        ]
        feature_df = feature_df[training_columns]
        
        return self.reward_simulator.predict(feature_df)[0]

    def step(self, action):
        self.current_step += 1
        x, y, width, height, num_elements = self.current_state

        # Assume a virtual container size for normalization
        container_width, container_height = 800, 600
        move_amount = config.MOVE_AMOUNT / container_width
        size_amount = config.SIZE_AMOUNT / container_width

        if action == 0: y -= move_amount
        elif action == 1: y += move_amount
        elif action == 2: x -= move_amount
        elif action == 3: x += move_amount
        elif action == 4: width += size_amount
        elif action == 5: width = max(0.05, width - size_amount)
        elif action == 6: height += size_amount
        elif action == 7: height = max(0.05, height - size_amount)
        
        self.current_state = np.array([x, y, width, height, num_elements], dtype=np.float32)

        if (x < 0 or y < 0 or (x + width) > 1 or (y + height) > 1):
            reward = -1.0
            done = True
        else:
            new_potential = self._get_potential(self.current_state)
            reward = new_potential - self.previous_potential
            self.previous_potential = new_potential
            done = self.current_step >= self.max_steps_per_episode

        return self.current_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.current_state = np.array([0.5, 0.5, 0.2, 0.1, 0.05], dtype=np.float32)
        self.previous_potential = self._get_potential(self.current_state)
        return self.current_state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    # Test the environment with a specific persona
    env = PagePilotEnv(persona='power_user')
    obs = env.reset()
    print(f"Testing with persona: {env.persona}")
    print(f"Initial State: {obs}")
    print(f"Initial Potential: {env.previous_potential:.4f}")

    for i in range(5):
        action = env.action_space.sample()
        action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'IncW', 5: 'DecW', 6: 'IncH', 7: 'DecH'}
        print(f"\n--- Step {i+1}, Action: {action_map[action]} ---")
        obs, reward, done, info = env.step(action)
        print(f"New State: {obs}")
        print(f"Reward: {reward:.4f}")
        if done:
            print("Episode finished.")
            obs = env.reset()
    env.close()

