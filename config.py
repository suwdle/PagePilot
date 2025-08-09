
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
MAX_STEPS_PER_EPISODE = 20

# --- DQN Trainer Settings ---
EPISODES = 50
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 10000
