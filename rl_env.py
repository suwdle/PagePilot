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

        # Load the dataset and the reward model
        self.df = pd.read_csv(data_path)
        self.reward_model = joblib.load(model_path)
        self.current_index = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example: discrete action space for moving an element up, down, left, or right
        self.action_space = spaces.Discrete(4) # 0: up, 1: down, 2: left, 3: right
        
        # Example: observation space based on the features from the CSV
        # The shape should match the number of features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(len(self.df.columns) - 1,), dtype=np.float32)

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Apply the action to the current state
        # For now, this is a placeholder. We'll implement the logic later.
        new_state_vector = self._take_action(action)

        # Get the reward for the new state
        reward = self._get_reward(new_state_vector)

        # For simplicity, we'll move to the next UI element in the dataset
        self.current_index += 1
        done = self.current_index >= len(self.df)

        # Get the next observation
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        self.current_index = 0
        return self._next_observation()

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen.
        """
        # For now, we'll just print the current state
        print(f'Current State: \n{self.df.iloc[self.current_index]}')
        print(f'Predicted CTR: {self._get_reward(self.df.iloc[self.current_index].drop("simulated_ctr").values)}')

    def _next_observation(self):
        """
        Get the data points for the next observation.
        """
        return self.df.iloc[self.current_index].drop('simulated_ctr').values

    def _take_action(self, action):
        """
        Apply the action to the current state vector.
        (Placeholder for now)
        """
        # This is a simplified placeholder. 
        # A real implementation would modify the state vector based on the action.
        state = self._next_observation()
        # For example, action 0 (up) might decrease the 'center_y' value
        if action == 0:
            state[6] -= 10 # Assuming center_y is at index 6
        return state

    def _get_reward(self, state_vector):
        """
        Get the reward for a given state vector.
        """
        # The reward is the predicted CTR from our trained model.
        # The model expects a 2D array, so we reshape.
        return self.reward_model.predict(state_vector.reshape(1, -1))[0]


def main():
    """
    Example of how to use the PagePilotEnv.
    """
    data_path = "/home/seokjun/pj/PagePilot/data/labeled_waveui.csv"
    model_path = "/home/seokjun/pj/PagePilot/models/reward_simulator_lr.joblib"
    
    env = PagePilotEnv(data_path, model_path)

    obs = env.reset()
    print("Initial Observation:")
    print(obs)

    action = env.action_space.sample() # Get a random action
    obs, reward, done, info = env.step(action)

    print(f"\nAction taken: {action}")
    print(f"New Observation:\n{obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

if __name__ == "__main__":
    main()
