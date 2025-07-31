import gym
from gym import spaces
import numpy as np
import yaml
from playwright.sync_api import sync_playwright
import time

class LivePagePilotEnv(gym.Env):
    """
    A custom Gym environment for live web UI optimization using Playwright.
    It interacts with a real web page in a browser.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config_path='config.yaml', max_elements=50, feature_size=5):
        super(LivePagePilotEnv, self).__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.target_url = self.config['target_url']
        self.goal_selector = self.config['goal_element_selector']
        self.optimization_objective = self.config['optimization_objective']
        self.max_steps = self.config.get('optimization_steps', 20)

        self.max_elements = max_elements
        self.feature_size = feature_size  # element_type, x, y, width, height

        # Action Space: [element_index, action_type]
        # action_type: 0-4 (move up/down/left/right), 5-6 (increase/decrease size)
        self.action_space = spaces.Discrete(self.max_elements * 7)

        # Observation Space: Padded list of element features (normalized)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.max_elements * self.feature_size,), dtype=np.float32
        )

        # Playwright setup
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False) # Headless=False for viz
        self.page = self.browser.new_page()
        self.page.goto(self.target_url, wait_until='load')

        self.state = []
        self.current_step = 0
        self.initial_goal_metric = 0

    def _get_state(self):
        """Scans the DOM to get the current state of all visible elements."""
        elements = self.page.query_selector_all('body *[style*="visibility: visible"] ')
        state_list = []
        viewport_size = self.page.viewport_size
        
        for el in elements[:self.max_elements]:
            try:
                box = el.bounding_box()
                if box and box['width'] > 0 and box['height'] > 0:
                    # Normalize features
                    features = {
                        'selector': el.evaluate('(e) => e.tagName'), # Using tagName as a simple selector
                        'x': box['x'] / viewport_size['width'],
                        'y': box['y'] / viewport_size['height'],
                        'width': box['width'] / viewport_size['width'],
                        'height': box['height'] / viewport_size['height']
                    }
                    state_list.append(features)
            except Exception:
                continue # Skip elements that are not visible or don't have a bounding box
        self.state = state_list
        return self._state_to_vector(self.state)

    def _state_to_vector(self, state_list):
        """Converts the list of element dicts into a fixed-size numpy vector."""
        vector = np.zeros((self.max_elements, self.feature_size), dtype=np.float32)
        for i, element in enumerate(state_list):
            if i >= self.max_elements:
                break
            # A simple hash for element type, can be improved
            vector[i, 0] = hash(element['selector']) % 100 / 100.0 
            vector[i, 1] = element['x']
            vector[i, 2] = element['y']
            vector[i, 3] = element['width']
            vector[i, 4] = element['height']
        return vector.flatten()

    def _get_reward(self):
        """Calculates the reward based on the optimization objective with detailed debugging."""
        try:
            goal_element = self.page.query_selector(self.goal_selector)
            if not goal_element:
                print(f"[DEBUG] Reward is -1: Goal element with selector '{self.goal_selector}' not found.")
                return -1 # Penalty if the goal element is not found

            box = goal_element.bounding_box()
            if not box:
                print(f"[DEBUG] Reward is -1: Goal element '{self.goal_selector}' found, but it has no bounding box (it might be invisible or have zero size).")
                return -1

            if self.optimization_objective == 'maximize_size':
                current_metric = box['width'] * box['height']
                if self.initial_goal_metric <= 0:
                    print("[DEBUG] Initial goal metric is 0 or less. Reward will be 0.")
                    return 0
                reward = (current_metric - self.initial_goal_metric) / self.initial_goal_metric
                print(f"[DEBUG] Maximize Size Reward: (current: {current_metric:.2f} - initial: {self.initial_goal_metric:.2f}) / initial = {reward:.4f}")
                return reward
            
            elif self.optimization_objective == 'move_to_center':
                viewport = self.page.viewport_size
                center_x = box['x'] + box['width'] / 2
                center_y = box['y'] + box['height'] / 2
                distance = np.sqrt((center_x - viewport['width']/2)**2 + (center_y - viewport['height']/2)**2)
                # Normalize distance to be between 0 and 1
                max_dist = np.sqrt((viewport['width']/2)**2 + (viewport['height']/2)**2)
                normalized_distance = distance / max_dist
                reward = 1 - normalized_distance # Higher reward for being closer to center
                print(f"[DEBUG] Move to Center Reward: 1 - (distance: {distance:.2f} / max_dist: {max_dist:.2f}) = {reward:.4f}")
                return reward
            else:
                print(f"[DEBUG] Unknown optimization_objective: {self.optimization_objective}")
                return 0
        except Exception as e:
            print(f"[DEBUG] Reward is -1: An unexpected error occurred in _get_reward: {e}")
            return -1 # Error penalty

    def _take_action(self, action):
        """Applies the chosen action to an element in the browser."""
        element_index = action // 7
        action_type = action % 7

        if element_index >= len(self.state):
            return # Invalid action

        selector = self.state[element_index]['selector']
        # This is a simplified and potentially fragile way to select elements.
        # A more robust method would generate unique selectors.
        js_script = f'''
            (selector) => {{
                const el = document.querySelectorAll(selector)[{element_index}];
                if (!el) return;

                el.style.position = 'relative'; // Ensure we can move it
                let top = parseInt(el.style.top) || 0;
                let left = parseInt(el.style.left) || 0;
                let width = el.offsetWidth;
                let height = el.offsetHeight;

                switch ({action_type}) {{
                    case 0: el.style.top = (top - 5) + 'px'; break; // Up
                    case 1: el.style.top = (top + 5) + 'px'; break; // Down
                    case 2: el.style.left = (left - 5) + 'px'; break; // Left
                    case 3: el.style.left = (left + 5) + 'px'; break; // Right
                    case 4: el.style.width = (width * 1.05) + 'px'; break; // Increase Width
                    case 5: el.style.height = (height * 1.05) + 'px'; break; // Increase Height
                    case 6: el.style.width = (width * 0.95) + 'px'; break; // Decrease Width
                    case 7: el.style.height = (height * 0.95) + 'px'; break; // Decrease Height
                }}
            }}
        '''
        try:
            self.page.evaluate(js_script, selector)
            time.sleep(0.1) # Wait for DOM to update
        except Exception as e:
            print(f"Could not apply action: {e}")

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        
        obs = self._get_state()
        reward = self._get_reward()
        
        done = self.current_step >= self.max_steps
        
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.page.goto(self.target_url, wait_until='load')
        time.sleep(1) # Allow page to settle
        
        # Set initial metric for reward calculation
        try:
            goal_element = self.page.query_selector(self.goal_selector)
            box = goal_element.bounding_box()
            if self.optimization_objective == 'maximize_size':
                self.initial_goal_metric = box['width'] * box['height']
        except Exception:
            self.initial_goal_metric = 0

        return self._get_state()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.page.screenshot()
        elif mode == 'human':
            # The browser window is already visible (headless=False)
            pass

    def close(self):
        self.browser.close()
        self.playwright.stop()

if __name__ == '__main__':
    # Example of using the environment
    env = LivePagePilotEnv()
    obs = env.reset()
    print(f"Initial state vector received, shape: {obs.shape}")

    for _ in range(5):
        action = env.action_space.sample() # Take a random action
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.4f}, Done: {done}")
        if done:
            break
    
    env.close()
    print("Environment closed.")
