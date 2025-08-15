

import gym
from gym import spaces
import numpy as np
import config

from enhanced_reward import EnhancedCTRGenerator

class PagePilotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, file_path=None): # file_path is no longer used
        super(PagePilotEnv, self).__init__()
        
        self.reward_generator = EnhancedCTRGenerator()
        self.action_space = spaces.Discrete(8)
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.container_size = {'width': 800, 'height': 600}
        self.current_step = 0
        self.max_steps_per_episode = config.MAX_STEPS_PER_EPISODE
        self.current_state = None
        self.previous_potential = 0

    def _get_potential(self, state):
        x, y, _, _, num_elements = state
        return self.reward_generator.generate_realistic_ctr(np.array([x, y, num_elements * 100]))

    def step(self, action):
        self.current_step += 1
        x, y, width, height, num_elements = self.current_state

        move_amount = config.MOVE_AMOUNT / self.container_size['width']
        size_amount = config.SIZE_AMOUNT / self.container_size['width']

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
        initial_x = 0.5
        initial_y = 0.5
        initial_width = 0.2
        initial_height = 0.1
        num_elements = 0.05 # 5 elements
        self.current_state = np.array([initial_x, initial_y, initial_width, initial_height, num_elements], dtype=np.float32)
        self.previous_potential = self._get_potential(self.current_state)
        return self.current_state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    env = PagePilotEnv()
    obs = env.reset()
    print(f"Initial State: {obs}")
    print(f"Initial Potential: {env.previous_potential:.4f}")

    for i in range(10):
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

