import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import config
from rl_env import PagePilotEnv
from dqn_trainer import DQN

st.set_page_config(layout="wide", page_title="PagePilot Dashboard", initial_sidebar_state="expanded")

# --- Style ---
plt.style.use('dark_background')

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    try:
        temp_env = PagePilotEnv()
        input_dim = temp_env.observation_space.shape[0]
        output_dim = temp_env.action_space.n
        temp_env.close()
        model = DQN(input_dim, output_dim).to(device)
        model.load_state_dict(torch.load(config.DQN_MODEL_PATH, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model not found at {config.DQN_MODEL_PATH}. Please run dqn_trainer.py first.")
        return None

# --- Visualization Function ---
def plot_ui_state(state, ax, title):
    ax.clear()
    ax.set_facecolor('#0e1117')
    container = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#4f8bf9', linewidth=2, linestyle='--')
    ax.add_patch(container)
    
    x, y, w, h, _ = state
    button = patches.Rectangle((x, y), w, h, facecolor='#4f8bf9', edgecolor='#99c0ff', linewidth=2)
    ax.add_patch(button)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, color='white')

# --- Streamlit UI ---
st.sidebar.title("⚙️ Controls")
persona = st.sidebar.selectbox("Select User Persona", ('casual_browser', 'power_user', 'elderly'))
run_button = st.sidebar.button("🚀 Run Optimization")

st.title("🤖 PagePilot - AI-Powered UI Optimization")
st.markdown("학습된 DQN 에이전트가 UI 레이아웃을 실시간으로 최적화하는 과정을 확인합니다. 사이드바에서 페르소나를 선택하고 버튼을 누르세요.")

policy_net = load_model()

if policy_net:
    # Placeholders for metrics and visuals
    col1, col2, col3 = st.columns(3)
    metric_placeholder1 = col1.empty()
    metric_placeholder2 = col2.empty()
    metric_placeholder3 = col3.empty()
    st.markdown("--- ")
    plot_placeholder = st.empty()
    log_expander = st.expander("Show Episode Log")
    log_placeholder = log_expander.empty()

    if run_button:
        env = PagePilotEnv(persona=persona)
        state = env.reset()
        total_reward = 0
        
        fig, ax = plt.subplots(figsize=(8, 8))
        log_text = ""

        for i in range(env.max_steps_per_episode):
            if i == 0:
                plot_ui_state(state, ax, f"Step {i}: Initial State")
                plot_placeholder.pyplot(fig)

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_id = policy_net(state_tensor).max(1)[1].item()

            state, reward, done, _ = env.step(action_id)
            total_reward += reward
            
            action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'IncW', 5: 'DecW', 6: 'IncH', 7: 'DecH'}
            action_name = action_map.get(action_id, "Unknown")

            # Update metrics
            metric_placeholder1.metric(label="Step", value=f"{i + 1}/{env.max_steps_per_episode}")
            metric_placeholder2.metric(label="Last Action", value=action_name)
            metric_placeholder3.metric(label="Total Reward", value=f"{total_reward:.4f}")

            # Update visualization
            step_info = f"Step {i+1}: Action = {action_name}, Reward = {reward:.4f}"
            plot_ui_state(state, ax, "Live Optimization")
            plot_placeholder.pyplot(fig)
            
            log_text += step_info + "\n"
            log_placeholder.text_area("Log", log_text, height=200)

            time.sleep(0.05)

            if done:
                st.success(f"✅ Optimization finished in {i+1} steps!")
                break
        
        env.close()
        plt.close(fig)
