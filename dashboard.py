import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import config
from rl_env import PagePilotGNNAEnv  # GNN 환경으로 변경
from gnn_agent import GNN_DQN         # GNN 에이전트로 변경

st.set_page_config(layout="wide", page_title="PagePilot GNN Dashboard", initial_sidebar_state="expanded")

# --- 스타일 설정 ---
plt.style.use('dark_background')

# --- 모델 로딩 ---
@st.cache_resource
def load_gnn_model():
    """학습된 GNN_DQN 모델을 로드합니다."""
    device = torch.device("cpu")
    try:
        temp_env = PagePilotGNNAEnv()
        node_feature_dim = temp_env.observation_space['x'].shape[1]
        num_actions_per_node = temp_env.action_space.nvec[1]
        temp_env.close()

        model = GNN_DQN(node_feature_dim, num_actions_per_node).to(device)
        model.load_state_dict(torch.load(config.DQN_MODEL_PATH, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"에러: {config.DQN_MODEL_PATH}에서 모델 파일을 찾을 수 없습니다. gnn_trainer.py를 먼저 실행하세요.")
        return None

# --- 시각화 함수 ---
def plot_graph_ui_state(state_graph, ax, title, highlighted_node=None):
    """그래프 기반 UI 상태를 시각화합니다."""
    ax.clear()
    ax.set_facecolor('#0e1117')
    container = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='#4f8bf9', linewidth=2, linestyle='--')
    ax.add_patch(container)
    
    nodes = state_graph.x.numpy()
    for i, node in enumerate(nodes):
        x, y, w, h, elem_type = node
        face_color = '#3498db' if elem_type == 0 else '#95a5a6'
        edge_color = '#99c0ff' if elem_type == 0 else '#ecf0f1'
        linewidth = 3 if highlighted_node is not None and i == highlighted_node else 1.5

        button = patches.Rectangle((x, y), w, h, facecolor=face_color, edgecolor=edge_color, linewidth=linewidth)
        ax.add_patch(button)
        ax.text(x + w/2, y + h/2, f"ID:{i}", ha='center', va='center', color='white', fontsize=10)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, color='white')

# --- Streamlit UI ---
st.sidebar.title("⚙️ GNN Controls")
persona = st.sidebar.selectbox("Select User Persona", ('casual_browser', 'power_user', 'elderly'))
num_elements = st.sidebar.slider("Number of UI Elements", 2, 10, 5)
run_button = st.sidebar.button("🚀 Run GNN Optimization")

st.title("🤖 PagePilot - GNN-Powered UI Optimization")
st.markdown("학습된 GNN 에이전트가 **여러 UI 요소**들의 관계를 파악하고 레이아웃을 실시간으로 최적화하는 과정을 확인합니다.")

policy_net = load_gnn_model()

if policy_net:
    col1, col2, col3 = st.columns(3)
    metric_placeholder1 = col1.empty()
    metric_placeholder2 = col2.empty()
    metric_placeholder3 = col3.empty()
    st.markdown("---")
    plot_placeholder = st.empty()
    log_expander = st.expander("Show Episode Log")
    log_placeholder = log_expander.empty()

    if run_button:
        env = PagePilotGNNAEnv(num_elements=num_elements, persona=persona)
        state_graph = env.reset()
        total_reward = 0
        
        fig, ax = plt.subplots(figsize=(8, 8))
        log_text = ""

        for i in range(env.max_steps_per_episode):
            if i == 0:
                plot_graph_ui_state(state_graph, ax, f"Step {i}: Initial State")
                plot_placeholder.pyplot(fig)

            with torch.no_grad():
                q_values = policy_net(state_graph)
                action_node = q_values.argmax() // env.action_space.nvec[1]
                action_move = q_values.argmax() % env.action_space.nvec[1]
                action = (action_node.item(), action_move.item())

            state_graph, reward, done, _ = env.step(action)
            total_reward += reward
            
            action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'IncW', 5: 'DecW', 6: 'IncH', 7: 'DecH'}
            action_name = action_map.get(action[1], "Unknown")
            
            metric_placeholder1.metric(label="Step", value=f"{i + 1}/{env.max_steps_per_episode}")
            metric_placeholder2.metric(label="Last Action", value=f"Elem {action[0]}: {action_name}")
            metric_placeholder3.metric(label="Total Reward", value=f"{total_reward:.4f}")

            step_info = f"Step {i+1}: Action on Element {action[0]} = {action_name}"
            plot_graph_ui_state(state_graph, ax, "Live Optimization", highlighted_node=action[0])
            plot_placeholder.pyplot(fig)
            
            log_text += step_info + f", Reward = {reward:.4f}\n"
            log_placeholder.text_area("Log", log_text, height=200)

            time.sleep(0.1)

            if done:
                st.success(f"✅ Optimization finished in {i+1} steps!")
                break
        
        env.close()
        plt.close(fig)
