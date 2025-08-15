import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the local website file
WEBSITE_PATH = os.path.join(ROOT_DIR, "local_web", "index.html")

# Path to the trained DQN model
DQN_MODEL_PATH = os.path.join(ROOT_DIR, "models", "dqn_model.pth")

# Directory to save screenshots
SCREENSHOT_DIR = os.path.join(ROOT_DIR, "results", "screenshots")

# --- RL Environment Settings ---
MAX_STEPS_PER_EPISODE = 100
# Action step sizes
MOVE_AMOUNT = 1  # in pixels
SIZE_AMOUNT = 1  # in pixels
# Minimum button dimensions
MIN_BUTTON_WIDTH = 10 # in pixels
MIN_BUTTON_HEIGHT = 10 # in pixels
# Reward function weights
REWARD_WEIGHTS = {
    "position": 0.4,
    "size": 0.4,
    "contrast": 0.2
}
# Reward scaling factor
REWARD_SCALE = 100.0


# --- DQN Trainer Settings ---
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
LEARNING_RATE = 5e-4
REPLAY_BUFFER_SIZE = 2000
