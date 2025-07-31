import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import subprocess
import time
import os
from PIL import Image

# Import the live environment
from rl_env import LivePagePilotEnv

st.set_page_config(
    page_title="PagePilot Live Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 PagePilot Live Dashboard")

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ("Live Agent Visualization", "Configuration", "Run Optimization"))

# --- Config Loading and Editing ---
@st.cache_data
def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def save_config(config_data):
    with open("config.yaml", 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

# --- Page Implementations ---

if page == "Live Agent Visualization":
    st.header("🎨 Live Agent Optimization Visualizer")
    st.write("Watch the RL agent modify a live web page to improve a target metric.")

    # Display current configuration
    st.subheader("Current Configuration")
    config = load_config()
    st.json(config)

    if st.button("Start Live Visualization"):
        st.info("Initializing live environment and starting visualization...")
        
        try:
            env = LivePagePilotEnv(config_path='config.yaml')
            
            col1, col2 = st.columns(2)
            col1.subheader("Initial State")
            col2.subheader("Optimized State")
            
            # Initial render
            initial_obs = env.reset()
            initial_screenshot = env.render(mode='rgb_array')
            initial_reward = env._get_reward() # Get initial reward

            col1.image(initial_screenshot, caption=f"Initial Page | Reward: {initial_reward:.4f}", use_column_width=True)
            
            # Optimization loop
            progress_bar = st.progress(0)
            for i in range(env.max_steps):
                action = env.action_space.sample() # Using random action for visualization
                obs, reward, done, info = env.step(action)
                progress_bar.progress((i + 1) / env.max_steps)
                time.sleep(0.2)
                if done:
                    break
            
            # Final render
            final_screenshot = env.render(mode='rgb_array')
            final_reward = env._get_reward()
            col2.image(final_screenshot, caption=f"Final Page | Reward: {final_reward:.4f}", use_column_width=True)

            st.success("Live visualization complete!")
            st.metric("Reward Improvement", f"{final_reward:.4f}", delta=f"{final_reward - initial_reward:.4f}")
            
            env.close()

        except Exception as e:
            st.error(f"An error occurred during visualization: {e}")
            # Ensure env is closed if it was created
            if 'env' in locals() and env:
                env.close()

elif page == "Configuration":
    st.header("⚙️ System Configuration")
    st.write("Modify the optimization parameters below. Changes will be saved to `config.yaml`.")
    
    config_data = load_config()
    
    edited_config = {}
    edited_config['target_url'] = st.text_input("Target URL", value=config_data['target_url'])
    edited_config['goal_element_selector'] = st.text_input("Goal Element (CSS Selector)", value=config_data['goal_element_selector'])
    edited_config['optimization_objective'] = st.selectbox("Optimization Objective", ["maximize_size", "move_to_center"], index=["maximize_size", "move_to_center"].index(config_data['optimization_objective']))
    edited_config['optimization_steps'] = st.slider("Optimization Steps", 1, 100, value=config_data['optimization_steps'])
    edited_config['output_dir'] = st.text_input("Output Directory", value=config_data['output_dir'])

    if st.button("Save Configuration"):
        save_config(edited_config)
        st.success("Configuration saved successfully!")

elif page == "Run Optimization":
    st.header("▶️ Run Full Optimization")
    st.write("This will run the full optimization process in the background using the settings from the Configuration page.")

    if st.button("Start Background Optimization"):
        st.info("Starting optimization process... This may take a while.")
        
        log_file = "results/optimization_run.log"
        command = f"uv run python main.py > {log_file} 2>&1"
        
        try:
            # Make sure the results directory exists
            os.makedirs("results", exist_ok=True)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            st.session_state['process_pid'] = process.pid

            st.success(f"Optimization process started with PID: {process.pid}. Check logs for progress.")
            st.code(command)
            st.warning("The browser window controlled by Playwright will appear on the system running this script.")

        except Exception as e:
            st.error(f"Failed to start the optimization process: {e}")