import streamlit as st
import pandas as pd
import json
from pathlib import Path
import subprocess
import time
import os
import torch

# Import components for visualization
from rl_env import PagePilotEnv
from dqn_trainer import DQN
from dashboard_utils import render_ui_from_elements, run_optimization_visualizer

st.set_page_config(
    page_title="PagePilot Dashboard",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ PagePilot Dashboard")

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ("Agent Visualization", "Run New Test", "Test Results", "Training Status"))

# --- Model and Environment Loading --- 
@st.cache_resource
def load_models_and_env():
    """Loads the DQN model and environment, caching them for performance."""
    data_path = "./data/raw_elements.csv"
    reward_model_path = "./models/reward_simulator_lgbm.joblib"
    dqn_model_path = "./models/dqn_model.pth"

    if not os.path.exists(dqn_model_path):
        st.error(f"DQN model not found at {dqn_model_path}. Please train the agent first.")
        return None, None

    try:
        env = PagePilotEnv(data_path, reward_model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        
        policy_net = DQN(input_dim, output_dim).to(device)
        policy_net.load_state_dict(torch.load(dqn_model_path, map_location=device))
        policy_net.eval()
        return env, policy_net
    except Exception as e:
        st.error(f"Failed to load models or environment: {e}")
        return None, None

# --- Common Functions ---
def load_test_results():
    results_path = Path("results/dummy_results.json")
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except json.JSONDecodeError:
            return pd.DataFrame()
    return pd.DataFrame()

# --- Page Implementations ---

if page == "Agent Visualization":
    st.header("🎨 Agent Optimization Visualizer")
    st.write("Watch the RL agent modify a UI layout to improve the predicted Click-Through Rate (CTR).")

    env, policy_net = load_models_and_env()

    if env and policy_net:
        if st.button("Start Visualization"):
            st.info("Running optimization process... Please wait.")
            
            # Placeholders for the two columns
            col1, col2 = st.columns(2)
            col1.subheader("Initial UI")
            col2.subheader("Optimized UI")
            initial_img_placeholder = col1.empty()
            optimized_img_placeholder = col2.empty()
            initial_reward_placeholder = col1.empty()
            optimized_reward_placeholder = col2.empty()
            progress_bar = st.progress(0)
            
            steps = 10
            initial_state, final_state = None, None
            initial_reward, final_reward = 0, 0

            # Run the visualization generator
            visualization_generator = run_optimization_visualizer(env, policy_net, steps=steps)

            # Initial step
            _, initial_state, initial_reward = next(visualization_generator)
            initial_img = render_ui_from_elements(initial_state)
            initial_img_placeholder.image(initial_img, caption=f"Initial Predicted CTR: {initial_reward:.4f}", use_column_width=True)
            initial_reward_placeholder.metric("Initial CTR", f"{initial_reward:.4f}")

            # Loop through optimization steps
            for i, (status, state, reward) in enumerate(visualization_generator):
                final_state = state
                final_reward = reward
                progress_bar.progress((i + 1) / steps)
                time.sleep(0.2) # Small delay for visual effect

            # Display final result
            if final_state:
                final_img = render_ui_from_elements(final_state)
                optimized_img_placeholder.image(final_img, caption=f"Final Predicted CTR: {final_reward:.4f}", use_column_width=True)
                optimized_reward_placeholder.metric("Optimized CTR", f"{final_reward:.4f}", delta=f"{final_reward - initial_reward:.4f}")

            st.success("Visualization complete!")

elif page == "Training Status":
    st.header("🚀 Training Status")
    st.write("This section shows the status of the training processes.")

    # Check for model files to indicate training completion
    dqn_model_path = "./models/dqn_model.pth"
    reward_model_path = "./models/reward_simulator_lgbm.joblib"

    if os.path.exists(reward_model_path):
        st.success("✅ Reward Simulator model is trained.")
    else:
        st.warning("⚠️ Reward Simulator model not found. Please run `uv run python reward_simulator_trainer.py`")

    if os.path.exists(dqn_model_path):
        st.success("✅ DQN Agent model is trained.")
    else:
        st.warning("⚠️ DQN Agent model not found. Please run `uv run python dqn_trainer.py`")

    st.info("For real-time metrics, a more advanced setup using logging libraries and file monitoring would be needed.")

elif page == "Test Results":
    st.header("📊 Test Results Summary")

    df = load_test_results()

    if not df.empty:
        st.subheader("Overall Test Statistics")
        total_tests = len(df)
        # Ensure 'success' column is boolean or convertible
        df['success'] = df['success'].astype(bool)
        success_rate = (df['success'].sum() / total_tests) * 100
        avg_duration = df['duration_seconds'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tests", total_tests)
        col2.metric("Success Rate", f"{success_rate:.2f}%")
        col3.metric("Avg. Duration (s)", f"{avg_duration:.2f}")

        st.subheader("Test Results Overview")
        st.dataframe(df[['scenario_name', 'success', 'duration_seconds', 'timestamp']])

        st.subheader("Detailed Test Report")
        # Use a more robust way to select tests if test_id is not unique
        if 'test_id' in df.columns:
            selected_test_id = st.selectbox("Select a test to see details", df['test_id'].unique())

            if selected_test_id:
                test_details = df[df['test_id'] == selected_test_id].iloc[0]
                st.write(f"**Scenario:** {test_details['scenario_name']}")
                st.write(f"**Timestamp:** {test_details['timestamp']}")
                st.write(f"**Success:** {'✅ Pass' if test_details['success'] else '❌ Fail'}")
                if not test_details['success'] and 'error_message' in test_details:
                    st.error(f"**Error:** {test_details['error_message']}")

                if 'steps' in test_details and test_details['steps']:
                    with st.expander("Show execution steps and screenshots"):
                        for step in test_details['steps']:
                            st.write(f"- **Step {step.get('step', 'N/A')}:** {step.get('action', 'No action recorded')}")
                        st.info("Screenshots are placeholders and not displayed in this demo.")
    else:
        st.warning("No test results found. Run a test to see the results here.")

elif page == "Run New Test":
    st.header("▶️ Run a New Test")

    # Define available scenarios
    # In a real application, this could be loaded from a config file
    scenarios = {
        "Login Test": "login_scenario",
        "Purchase Item": "purchase_scenario",
        "Search Functionality": "search_scenario"
    }
    selected_scenario_name = st.selectbox("Select a scenario to run", list(scenarios.keys()))

    if st.button(f"Start '{selected_scenario_name}' Test"):
        scenario_id = scenarios[selected_scenario_name]
        st.info(f"Starting test for scenario: **{selected_scenario_name}**...")

        # Using subprocess to run the main agent script in the background
        # This is a simplified approach. For robust applications, a task queue like Celery is recommended.
        log_file = f"results/{scenario_id}_run.log"
        command = f"uv run python main.py --scenario {scenario_id} > {log_file} 2>&1"

        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            st.session_state['process_pid'] = process.pid

            st.success(f"Test process started with PID: {process.pid}. Check logs for progress.")
            st.code(command)

            with st.spinner("Test in progress... You can navigate to other pages."):
                # This just waits for a fixed time in this example.
                # A real implementation would monitor the process or a status file.
                time.sleep(10) # Simulate running time

            st.info("The test process is running in the background. Results will appear in the 'Test Results' tab upon completion.")

        except Exception as e:
            st.error(f"Failed to start the test process: {e}")
