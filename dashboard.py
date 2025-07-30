import streamlit as st

st.set_page_config(
    page_title="PagePilot Dashboard",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ PagePilot Dashboard")

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ("Training Status", "Test Results", "Run New Test"))

if page == "Training Status":
    st.header("Training Status")
    st.write("This section will show the real-time status of the RL agent training.")

elif page == "Test Results":
    st.header("Test Results")
    st.write("This section will display the results of completed tests.")

elif page == "Run New Test":
    st.header("Run a New Test")
    st.write("This section will allow you to run a new test with the trained agent.")
