import gym
from gym import spaces
import numpy as np
import pandas as pd
import joblib
import random

# Helper function from reward_simulator_trainer to create features dynamically
def create_aggregated_features(elements_df):
    """
    Dynamically creates aggregated features from a DataFrame of UI elements.
    """
    if elements_df.empty:
        # Return a zeroed DataFrame with the expected columns if the input is empty
        # This is a simplified example; you might need a more robust way to get all possible feature columns
        # A potential approach is to define the columns explicitly
        return pd.DataFrame(np.zeros((1, 10)), columns=[f'feature_{i}' for i in range(10)])


    df = elements_df.copy()
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['area'] = df['width'] * df['height']
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    df['text_density'] = df['ocr_text'].str.len() / (df['area'] + 1e-6)
    df['text_density'] = df['text_density'].fillna(0)

    df_type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, df_type_dummies], axis=1)

    agg_dict = {
        'width': 'mean', 'height': 'mean', 'area': ['mean', 'sum'],
        'center_x': 'mean', 'center_y': 'mean', 'text_density': 'mean',
        **{col: 'sum' for col in df_type_dummies.columns}
    }
    
    # Aggregate
    agg_df = pd.DataFrame(df.agg(agg_dict)).transpose()
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Add component count
    agg_df['component_count'] = len(df)
    
    return agg_df

class PagePilotEnv(gym.Env):
    """
    Custom Environment for UI layout optimization using raw element data.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, model_path, max_elements=50, feature_size=8):
        super(PagePilotEnv, self).__init__()

        self.raw_data = pd.read_csv(data_path)
        self.reward_model = joblib.load(model_path)
        self.unique_screenshot_ids = self.raw_data['screenshot_id'].unique()
        
        self.max_elements = max_elements
        self.feature_size = feature_size  # type, x1, y1, x2, y2, ocr_text_len, purpose_len
        
        # Action: 0-index of element, 1-property to change (e.g., x1), 2-new value
        # Simplified: select element, move it (up, down, left, right)
        self.action_space = spaces.Discrete(self.max_elements * 4) # 4 directions for each element
        
        # Observation: Padded list of element features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.max_elements * self.feature_size,), dtype=np.float32
        )
        
        self.state = [] # List of element dictionaries
        self.current_screenshot_id = None

    def _state_to_vector(self, state):
        """Converts the list of element dicts into a fixed-size numpy vector."""
        vector = np.zeros((self.max_elements, self.feature_size), dtype=np.float32)
        
        # Define a mapping from element type to a numerical value
        type_mapping = {t: i for i, t in enumerate(self.raw_data['type'].unique())}

        for i, element in enumerate(state):
            if i >= self.max_elements:
                break
            
            # Normalize features to be roughly between 0 and 1
            vector[i, 0] = type_mapping.get(element['type'], -1) / len(type_mapping)
            vector[i, 1] = element['x1'] / 1920  # Assuming max screen width
            vector[i, 2] = element['y1'] / 1080  # Assuming max screen height
            vector[i, 3] = element['x2'] / 1920
            vector[i, 4] = element['y2'] / 1080
            vector[i, 5] = len(element.get('ocr_text', '')) / 100 # Normalize text length
            vector[i, 6] = len(element.get('purpose', '')) / 50 # Normalize purpose length
            # Last feature can be padding or another metric
            vector[i, 7] = 1.0 # Indicates an active element

        return vector.flatten()

    def step(self, action):
        element_index = action // 4
        move_direction = action % 4

        if element_index >= len(self.state):
            # Invalid action, no change, small penalty
            reward = -0.1
            done = False
            return self._state_to_vector(self.state), reward, done, {"error": "Invalid element index"}

        self._take_action(element_index, move_direction)
        
        reward = self._get_reward()
        
        # For simplicity, we'll say an episode is one step. Can be changed.
        done = True 
        
        return self._state_to_vector(self.state), reward, done, {}

    def reset(self):
        self.current_screenshot_id = random.choice(self.unique_screenshot_ids)
        self.state = self.raw_data[self.raw_data['screenshot_id'] == self.current_screenshot_id].to_dict('records')
        return self._state_to_vector(self.state)

    def render(self, mode='human', close=False):
        print(f"Screenshot ID: {self.current_screenshot_id}")
        for element in self.state:
            print(element)

    def _take_action(self, element_index, move_direction):
        """Moves the selected element."""
        element = self.state[element_index]
        move_amount = 10 # pixels
        
        width = element['x2'] - element['x1']
        height = element['y2'] - element['y1']

        if move_direction == 0: # Up
            element['y1'] -= move_amount
            element['y2'] = element['y1'] + height
        elif move_direction == 1: # Down
            element['y1'] += move_amount
            element['y2'] = element['y1'] + height
        elif move_direction == 2: # Left
            element['x1'] -= move_amount
            element['x2'] = element['x1'] + width
        elif move_direction == 3: # Right
            element['x1'] += move_amount
            element['x2'] = element['x1'] + width
            
        # Update the state
        self.state[element_index] = element

    def _get_reward(self):
        """Calculates reward based on the current state's aggregated features."""
        if not self.state:
            return 0.0
            
        # Convert current state (list of dicts) to a DataFrame
        current_elements_df = pd.DataFrame(self.state)
        
        # Create aggregated features for the reward model
        aggregated_features = create_aggregated_features(current_elements_df)
        
        # The reward model expects features in a specific order.
        # We need to ensure the columns match what the model was trained on.
        # This is a critical step. For now, we assume the columns are compatible.
        # A robust implementation would save the feature list with the model.
        
        # Reorder columns to match the model's training columns
        # This is a simplified approach. A better way is to save/load the column order.
        try:
            reward_features = aggregated_features[self.reward_model.feature_name_]
            return self.reward_model.predict(reward_features)[0]
        except Exception as e:
            # Fallback if feature names don't align perfectly
            print(f"Warning: Could not align features for reward prediction. Error: {e}")
            # Providing a generic reward or handling the error appropriately
            return 0.0


def main():
    data_path = "./data/raw_elements.csv"
    model_path = "./models/reward_simulator_lgbm.joblib"
    env = PagePilotEnv(data_path, model_path)

    obs = env.reset()
    print("Initial Observation Vector (flattened):")
    print(obs)

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print(f"
Action taken: {action}")
    print(f"New Observation Vector (flattened):
{obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

if __name__ == "__main__":
    main()