import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from typing import Any

# Define the DQN Network
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

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_dqn_live(env: Any, config: dict):
    """
    Trains a DQN agent in a live environment.
    This function is adapted for the LivePagePilotEnv.
    """
    # Hyperparameters from config or defaults
    EPISODES = config.get('episodes', 1)
    BATCH_SIZE = config.get('batch_size', 32)
    GAMMA = config.get('gamma', 0.99)
    EPS_START = config.get('eps_start', 0.9)
    EPS_END = config.get('eps_end', 0.05)
    EPS_DECAY = config.get('eps_decay', 200)
    TARGET_UPDATE = config.get('target_update', 10)
    LEARNING_RATE = config.get('learning_rate', 1e-4)
    REPLAY_BUFFER_SIZE = config.get('replay_buffer_size', 10000)
    MODEL_SAVE_PATH = './models/dqn_live_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        policy_net.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

    steps_done = 0

    for i_episode in range(EPISODES):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for t in range(env.max_steps):
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

            next_state_vector, reward_val, done, _ = env.step(action.item())
            reward = torch.tensor([reward_val], device=device)
            total_reward += reward.item()

            next_state = torch.tensor(next_state_vector, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, reward, next_state, done)

            state = next_state

            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))
                
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch[3] if s is not None])
                
                state_batch = torch.cat(batch[0])
                action_batch = torch.cat(batch[1])
                reward_batch = torch.cat(batch[2])
                
                q_values = policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                if non_final_next_states.numel() > 0:
                    next_state_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions).squeeze().detach()
                
                expected_q_values = (next_state_values * GAMMA) + reward_batch.float()

                loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            print(f"Step {t+1}/{env.max_steps}, Action: {action.item()}, Reward: {reward_val:.4f}, Total Reward: {total_reward:.4f}")

            if done:
                break
        
        print(f"Episode {i_episode+1} finished with Total Reward: {total_reward:.2f}")

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Live optimization training complete.")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
