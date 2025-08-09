import streamlit as st
import torch
import os
import time
import config

from rl_env import PagePilotEnv
from dqn_trainer import DQN

st.set_page_config(
    page_title="PagePilot Live Optimization",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ PagePilot: Live UI Optimization")
st.write("Watch the trained RL agent optimize a local webpage. The agent's goal is to move the button to the center of the dashed container.")

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained DQN model. The dimensions are fixed for our env."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The dimensions are known and fixed for our specific environment
    input_dim = 4  # x, y, width, height
    output_dim = 8 # up, down, left, right, increase/decrease width/height
    
    model = DQN(input_dim, output_dim).to(device)
    if not os.path.exists(model_path):
        return None # Model will be checked for existence later
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_optimization_process(policy_net):
    """Creates an environment, runs the optimization, and closes the env."""
    env = PagePilotEnv(file_path=config.WEBSITE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.info("Starting optimization... The agent will now take control.")
    
    col1, col2 = st.columns(2)
    col1.subheader("Initial State")
    col2.subheader("Optimized State")

    initial_img_placeholder = col1.empty()
    initial_reward_placeholder = col1.empty()
    optimized_img_placeholder = col2.empty()
    optimized_reward_placeholder = col2.empty()
    
    try:
        # Initial State
        initial_state = env.reset()
        initial_reward = env._calculate_reward(initial_state)
        initial_screenshot_path = os.path.join(config.SCREENSHOT_DIR, "initial_state.png")
        env.render(output_path=initial_screenshot_path)
        
        initial_img_placeholder.image(initial_screenshot_path, caption=f"Initial UI")
        initial_reward_placeholder.metric(label="Initial Reward", value=f"{initial_reward:.4f}")

        state = initial_state

        # Optimization Loop
        for i in range(config.MAX_STEPS_PER_EPISODE):
            time.sleep(0.5)

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_tensor).max(1)[1].item()
            
            state, reward, done, _ = env.step(action)

            screenshot_path = os.path.join(config.SCREENSHOT_DIR, f"step_{i+1}.png")
            env.render(output_path=screenshot_path)
            
            optimized_img_placeholder.image(screenshot_path, caption=f"Step {i+1}")
            delta_reward = reward - initial_reward
            optimized_reward_placeholder.metric(label=f"Step {i+1} Reward", value=f"{reward:.4f}", delta=f"{delta_reward:.4f}")

            if done:
                break
        
        st.success("Optimization episode finished!")

    except Exception as e:
        st.error(f"An error occurred during optimization: {e}")
    finally:
        st.info("Closing environment.")
        env.close()

# --- Main Application ---
policy_net = load_model(config.DQN_MODEL_PATH)

if not policy_net:
    st.error(f"DQN model not found at {config.DQN_MODEL_PATH}. Please train the agent first by running `uv run python dqn_trainer.py`.")
else:
    if st.button("🚀 Start Optimization Process"):
        run_optimization_process(policy_net)