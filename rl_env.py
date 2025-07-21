import gym
from gym import spaces
import numpy as np
import pandas as pd
import joblib

class PagePilotEnv(gym.Env):
    """
    Custom Environment for UI layout optimization that follows the OpenAI Gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, model_path):
        super(PagePilotEnv, self).__init__()

        self.df = pd.read_csv(data_path)
        self.reward_model = joblib.load(model_path)
        self.current_index = 0
        self.state_columns = [col for col in self.df.columns if col != 'simulated_ctr']

        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.state_columns),), dtype=np.float32)

    def step(self, action):
        current_state_series = self.df.loc[self.current_index, self.state_columns]
        new_state_series = self._take_action(action, current_state_series)
        reward = self._get_reward(new_state_series)

        self.current_index += 1
        done = self.current_index >= len(self.df)

        if not done:
            next_observation = self.df.loc[self.current_index, self.state_columns]
        else:
            next_observation = pd.Series(np.zeros(len(self.state_columns)), index=self.state_columns)

        return next_observation.values.astype(np.float32), reward, done, {}

    def reset(self):
        self.current_index = 0
        return self.df.loc[self.current_index, self.state_columns].values.astype(np.float32)

    def render(self, mode='human', close=False):
        print(f'Current State: \n{self.df.iloc[self.current_index]}')
        print(f'Predicted CTR: {self._get_reward(self.df.loc[self.current_index, self.state_columns])}')

    def _take_action(self, action, state_series):
        new_state = state_series.copy()
        if action == 0:  # Up
            new_state['center_y'] -= 10
        elif action == 1:  # Down
            new_state['center_y'] += 10
        elif action == 2:  # Left
            new_state['center_x'] -= 10
        elif action == 3:  # Right
            new_state['center_x'] += 10
        return new_state

    def _get_reward(self, state_series):
        return self.reward_model.predict(state_series.values.reshape(1, -1))[0]

def main():
    data_path = "./data/labeled_waveui.csv"
    model_path = "./models/reward_simulator_lr.joblib"
    env = PagePilotEnv(data_path, model_path)

    obs = env.reset()
    print("Initial Observation:")
    print(obs)

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print(f"\nAction taken: {action}")
    print(f"New Observation:\n{obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

if __name__ == "__main__":
    main()