import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import config

from rl_env import PagePilotEnv

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        if next_state is not None:
            next_state = torch.tensor(next_state, dtype=torch.float32)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_dqn():
    EPISODES = config.EPISODES
    BATCH_SIZE = config.BATCH_SIZE
    GAMMA = config.GAMMA
    EPS_START = config.EPS_START
    EPS_END = config.EPS_END
    EPS_DECAY = config.EPS_DECAY
    TARGET_UPDATE = config.TARGET_UPDATE
    LEARNING_RATE = config.LEARNING_RATE
    REPLAY_BUFFER_SIZE = config.REPLAY_BUFFER_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize here once to get dimensions
    temp_env = PagePilotEnv(persona='casual_browser')
    input_dim = temp_env.observation_space.shape[0]
    output_dim = temp_env.action_space.n
    temp_env.close()

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
    steps_done = 0

    for i_episode in range(EPISODES):
        # Create a new environment for each episode with a specific persona
        env = PagePilotEnv(persona='casual_browser')
        try:
            state = env.reset()
            total_reward = 0

            for t in range(env.max_steps_per_episode):
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                np.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1
                
                if random.random() > eps_threshold:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = policy_net(state_tensor).max(1)[1].item()
                else:
                    action = env.action_space.sample()

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                memory.push(state, action, reward, next_state if not done else None, done)
                state = next_state

                if len(memory) > BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = list(zip(*transitions))
                    state_batch = torch.stack(batch[0]).to(device)
                    action_batch = torch.stack(batch[1]).to(device)
                    reward_batch = torch.stack(batch[2]).to(device)
                    
                    non_final_mask = torch.tensor(tuple(s is not None for s in batch[3]), device=device, dtype=torch.bool)
                    non_final_next_states_list = [s for s in batch[3] if s is not None]
                    
                    if len(non_final_next_states_list) > 0:
                        non_final_next_states = torch.stack(non_final_next_states_list).to(device)
                        next_state_values = torch.zeros(BATCH_SIZE, device=device)
                        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                        expected_q_values = (next_state_values * GAMMA) + reward_batch.squeeze()
                    else:
                        expected_q_values = reward_batch.squeeze()

                    q_values = policy_net(state_batch).gather(1, action_batch)
                    loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
                    optimizer.step()

                if done:
                    break
            
            print(f"Episode {i_episode+1}, Total Reward: {total_reward:.2f}")

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        finally:
            env.close()

    os.makedirs(os.path.dirname(config.DQN_MODEL_PATH), exist_ok=True)
    torch.save(policy_net.state_dict(), config.DQN_MODEL_PATH)
    print(f"Model saved to {config.DQN_MODEL_PATH}")

if __name__ == "__main__":
    train_dqn()
