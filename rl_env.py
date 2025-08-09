

import gym
from gym import spaces
import numpy as np
from playwright.sync_api import sync_playwright
import os
import config

class PagePilotEnv(gym.Env):
    """
    Custom Environment for UI layout optimization that interacts with a local website
    using Playwright.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, file_path):
        super(PagePilotEnv, self).__init__()

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()

        self.local_file_path = "file://" + os.path.abspath(file_path)
        self.page.goto(self.local_file_path)

        # Define the action space: 0:Up, 1:Down, 2:Left, 3:Right
        self.action_space = spaces.Discrete(4)

        # Define the observation space: [x, y, width, height] of the button
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([2000, 2000, 500, 500]), # Generous limits
            dtype=np.float32
        )

        # Get container dimensions for reward calculation
        container_box = self.page.locator("#element-container").bounding_box()
        self.container_center_x = container_box['x'] + container_box['width'] / 2
        self.container_center_y = container_box['y'] + container_box['height'] / 2
        self.max_distance = np.sqrt(self.container_center_x**2 + self.container_center_y**2)

        self.current_step = 0
        self.max_steps_per_episode = config.MAX_STEPS_PER_EPISODE

    def _get_state(self):
        """Retrieves the current state (button's bounding box) from the webpage."""
        button_box = self.page.locator("#cta-button").bounding_box()
        if button_box:
            return np.array([
                button_box['x'],
                button_box['y'],
                button_box['width'],
                button_box['height']
            ], dtype=np.float32)
        # Return a zero vector if the element is not found
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _calculate_reward(self, state):
        """Calculates reward based on the button's position."""
        button_center_x = state[0] + state[2] / 2
        button_center_y = state[1] + state[3] / 2

        # Calculate distance from the container's center
        distance = np.sqrt(
            (button_center_x - self.container_center_x)**2 +
            (button_center_y - self.container_center_y)**2
        )

        # Normalize distance and invert it to get a proximity score
        # Reward is higher when the button is closer to the center
        reward = 1.0 - (distance / self.max_distance)
        return reward

    def step(self, action):
        self.current_step += 1

        # --- Take Action ---
        move_amount = 10  # pixels to move
        button_id = "#cta-button"

        # Get current position
        button_style = self.page.evaluate(f"window.getComputedStyle(document.querySelector('{button_id}'))")
        top = float(button_style['top'].replace('px', ''))
        left = float(button_style['left'].replace('px', ''))

        if action == 0:  # Up
            top -= move_amount
        elif action == 1:  # Down
            top += move_amount
        elif action == 2:  # Left
            left -= move_amount
        elif action == 3:  # Right
            left += move_amount

        # Apply the new style using JavaScript
        self.page.evaluate(f"document.querySelector('{button_id}').style.top = '{top}px';")
        self.page.evaluate(f"document.querySelector('{button_id}').style.left = '{left}px';")

        # --- Get New State and Reward ---
        new_state = self._get_state()
        reward = self._calculate_reward(new_state)

        # --- Check if Done ---
        done = self.current_step >= self.max_steps_per_episode

        return new_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        # Reload the page to reset the button's position to its initial state
        self.page.goto(self.local_file_path)
        return self._get_state()

    def render(self, mode='human', output_path="results/screenshots/step.png"):
        if mode == 'rgb_array':
            return self.page.screenshot()
        elif mode == 'human':
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.page.screenshot(path=output_path)
            print(f"Screenshot saved to {output_path}")

    def close(self):
        """Closes the Playwright browser."""
        self.browser.close()
        self.playwright.stop()

if __name__ == '__main__':
    # --- Test the new environment ---
    env = PagePilotEnv(file_path=config.WEBSITE_PATH)
    
    obs = env.reset()
    print(f"Initial State (Observation): {obs}")

    for i in range(5):
        action = env.action_space.sample()
        print(f"\n--- Step {i+1} ---")
        print(f"Action taken: {['Up', 'Down', 'Left', 'Right'][action]}")
        
        obs, reward, done, info = env.step(action)
        
        print(f"New State: {obs}")
        print(f"Reward: {reward:.4f}")
        
        # Render a screenshot
        screenshot_path = os.path.join(config.SCREENSHOT_DIR, f"step_{i+1}.png")
        env.render(mode='human', output_path=screenshot_path)

        if done:
            print("Episode finished.")
            break
            
    env.close()
    print("\nEnvironment test finished and browser closed.")
