# PagePilot: Reinforcement Learning for UI/UX Optimization

This project aims to build a reinforcement learning (RL) agent that can automatically optimize UI layouts to improve user engagement, measured by metrics like Click-Through Rate (CTR) and dwell time.

## Project Overview

The core idea is to train an RL agent to make sequential adjustments to UI elements (e.g., changing the position, size, or color of buttons). The agent's goal is to maximize a reward signal, which is derived from a simulated user response model.

### Key Components

- **Dataset**: We use the `agentsea/wave-ui-25k` dataset, which contains over 25,000 UI layout examples with detailed annotations for each UI element.
- **Reward Simulator**: Since we don't have real user interaction data, we build a reward simulator. This model takes a UI layout as input and predicts a simulated CTR. It is trained on heuristically generated labels based on established UI/UX design principles (e.g., larger buttons in the center of the screen are more likely to be clicked).
- **Reinforcement Learning Environment**: We have developed a custom environment following the OpenAI Gym interface. The agent's state is represented by a feature vector of the UI layout, actions are discrete adjustments to UI elements, and the reward is the predicted CTR from our simulator.
- **RL Agent**: A Deep Q-Network (DQN) agent is being trained to learn the optimal policy for modifying UI layouts to maximize the expected CTR.

## Current Status

- [X] **Phase 1 & 2**: Data preprocessing, feature engineering, and reward simulator training are complete.
- [X] **Phase 3**: The custom RL environment (`PagePilotEnv`) is implemented and tested.
- [ ] **Phase 4**: DQN agent training is currently in progress.
- [ ] **Phase 5**: Evaluation and visualization of the trained agent's performance.

This project provides a framework for applying RL to UI/UX design, with the potential to automate and discover non-obvious design improvements.