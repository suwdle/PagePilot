

import gym
from gym import spaces
import numpy as np
from playwright.sync_api import sync_playwright
import os
import config

import re

def get_color_as_rgb(color_str):
    """Converts a CSS color string (e.g., 'rgb(0, 123, 255)' or 'rgba(0,0,0,0)') to an (r, g, b) tuple."""
    # print(f"DEBUG: get_color_as_rgb received: {color_str}") # Keep for now, remove later
    if not color_str or ('rgb' not in color_str and 'rgba' not in color_str):
        return (255, 255, 255) # Default to white if color is not found
    try:
        # Extract all numbers, then take only the first three (r, g, b)
        numbers = list(map(int, re.findall(r'\d+', color_str)))
        return tuple(numbers[:3])
    except (ValueError, TypeError):
        return (255, 255, 255) # Default to white on error

def get_relative_luminance(rgb):
    """Calculates the relative luminance for an RGB color."""
    r, g, b = [x / 255.0 for x in rgb]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrast_ratio(rgb1, rgb2):
    """Calculates the contrast ratio between two RGB colors."""
    l1 = get_relative_luminance(rgb1)
    l2 = get_relative_luminance(rgb2)
    if l1 > l2:
        return (l1 + 0.05) / (l2 + 0.05)
    else:
        return (l2 + 0.05) / (l1 + 0.05)

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

        # Define the action space: 0-3 for movement, 4-7 for size changes
        self.action_space = spaces.Discrete(8)

        # Define the observation space: [x, y, width, height] of the button
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([2000, 2000, 500, 500]), # Generous limits
            dtype=np.float32
        )

        # Get container dimensions for reward calculation
        self.container_box = self.page.locator("#element-container").bounding_box()
        self.container_center_x = self.container_box['x'] + self.container_box['width'] / 2
        self.container_center_y = self.container_box['y'] + self.container_box['height'] / 2
        self.max_distance = np.sqrt(self.container_center_x**2 + self.container_center_y**2)
        
        # Get container color for contrast calculation
        container_style = self.page.locator("#element-container").evaluate("el => window.getComputedStyle(el)")
        self.container_color_rgb = get_color_as_rgb(container_style['backgroundColor'])


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
        """Calculates a heuristic CTR based on size, position, and contrast."""
        x, y, width, height = state

        # --- 1. Position Reward (closer to center is better) ---
        button_center_x = x + width / 2
        button_center_y = y + height / 2
        distance = np.sqrt(
            (button_center_x - self.container_center_x)**2 +
            (button_center_y - self.container_center_y)**2
        )
        position_reward = 1.0 - (distance / self.max_distance)

        # --- 2. Size Reward (larger is better, up to a limit) ---
        max_area = self.container_box['width'] * self.container_box['height']
        button_area = width * height
        size_reward = button_area / max_area

        # --- 3. Contrast Reward (higher contrast is better) ---
        button_style = self.page.locator("#cta-button").evaluate("el => window.getComputedStyle(el)")
        button_color_rgb = get_color_as_rgb(button_style['backgroundColor'])
        contrast_ratio = get_contrast_ratio(button_color_rgb, self.container_color_rgb)
        # Normalize contrast reward (WCAG AA requires 4.5:1)
        contrast_reward = min(contrast_ratio / 4.5, 1.0)

        # --- 4. Boundary Penalty (large penalty for being outside) ---
        penalty = 0
        if (x < self.container_box['x'] or 
            y < self.container_box['y'] or 
            x + width > self.container_box['x'] + self.container_box['width'] or 
            y + height > self.container_box['y'] + self.container_box['height']):
            penalty = -1.0 # Severe penalty

        if penalty < 0:
            return penalty

        # --- Weighted Combination ---
        # Weights can be tuned
        w_pos = 0.4
        w_size = 0.4
        w_contrast = 0.2

        total_reward = (w_pos * position_reward) + (w_size * size_reward) + (w_contrast * contrast_reward)
        return total_reward

    def step(self, action):
        self.current_step += 1
        button_id = "#cta-button"

        # --- Take Action ---
        # Get current style
        style = self.page.evaluate(f"window.getComputedStyle(document.querySelector('{button_id}'))")
        top = float(style['top'].replace('px', ''))
        left = float(style['left'].replace('px', ''))
        width = float(style['width'].replace('px', ''))
        height = float(style['height'].replace('px', ''))

        move_amount = 5  # pixels
        size_amount = 5  # pixels

        if action == 0:  # Up
            top -= move_amount
        elif action == 1:  # Down
            top += move_amount
        elif action == 2:  # Left
            left -= move_amount
        elif action == 3:  # Right
            left += move_amount
        elif action == 4:  # Increase Width
            width += size_amount
        elif action == 5:  # Decrease Width
            width = max(10, width - size_amount) # Min width 10px
        elif action == 6:  # Increase Height
            height += size_amount
        elif action == 7:  # Decrease Height
            height = max(10, height - size_amount) # Min height 10px

        # Apply the new style using JavaScript
        self.page.evaluate(f"""
            const btn = document.querySelector('{button_id}');
            btn.style.top = '{top}px';
            btn.style.left = '{left}px';
            btn.style.width = '{width}px';
            btn.style.height = '{height}px';
        """)

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
        action_map = {
            0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right',
            4: 'Increase Width', 5: 'Decrease Width', 6: 'Increase Height', 7: 'Decrease Height'
        }
        print(f"\n--- Step {i+1} ---")
        print(f"Action taken: {action_map[action]}")
        
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
