import streamlit as st
import pandas as pd
import json
from pathlib import Path
import subprocess
import time

st.set_page_config(
    page_title="PagePilot Dashboard",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ PagePilot Dashboard")

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ("Run New Test", "Test Results", "Training Status"))

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

if page == "Training Status":
    st.header("🚀 Training Status")
    st.write("This section will show the real-time status of the RL agent training.")
    # Placeholder for future implementation
    st.info("Real-time training metrics will be displayed here.")

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
