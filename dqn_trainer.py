

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from rl_env import PagePilotEnv

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

# Main Training Function
def train_dqn():
    # Hyperparameters
    EPISODES = 100
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    LEARNING_RATE = 1e-4
    REPLAY_BUFFER_SIZE = 10000

    # Initialize environment and networks
    data_path = "/home/seokjun/pj/PagePilot/data/labeled_waveui.csv"
    model_path = "/home/seokjun/pj/PagePilot/models/reward_simulator_lr.joblib"
    env = PagePilotEnv(data_path, model_path)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

    steps_done = 0

    for i_episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            # Select and perform an action
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state)).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            memory.push(state, action, reward, next_state, done)

            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))

                state_batch = torch.FloatTensor(np.array(batch[0]))
                action_batch = torch.LongTensor(batch[1]).unsqueeze(1)
                reward_batch = torch.FloatTensor(batch[2])
                next_state_batch = torch.FloatTensor(np.array(batch[3]))
                done_batch = torch.BoolTensor(batch[4])

                # Compute Q(s_t, a)
                q_values = policy_net(state_batch).gather(1, action_batch)

                # Compute V(s_{t+1})
                next_q_values = target_net(next_state_batch).max(1)[0].detach()
                expected_q_values = reward_batch + (GAMMA * next_q_values * ~done_batch)

                # Compute Huber loss
                loss = nn.functional.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if done:
                break
        
        print(f"Episode {i_episode+1}, Total Reward: {total_reward:.2f}")

        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Training complete.")
    # Save the trained model
    torch.save(policy_net.state_dict(), '/home/seokjun/pj/PagePilot/models/dqn_model.pth')
    print("Model saved.")

if __name__ == "__main__":
    train_dqn()
