import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import config
from rl_env import PagePilotGNNAEnv  # GNN 환경 임포트
from gnn_agent import GNN_DQN       # GNN 모델 임포트

class ReplayBuffer:
    """GNN 환경을 위한 리플레이 버퍼"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # torch.tensor로 변환하지 않고, Data 객체 그대로 저장
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_gnn_dqn():
    """GNN 에이전트를 학습시키는 메인 함수"""
    # --- 하이퍼파라미터 ---
    EPISODES = config.EPISODES
    BATCH_SIZE = config.BATCH_SIZE
    GAMMA = config.GAMMA
    EPS_START = config.EPS_START
    EPS_END = config.EPS_END
    EPS_DECAY = config.EPS_DECAY
    TARGET_UPDATE = config.TARGET_UPDATE
    LEARNING_RATE = config.LEARNING_RATE
    REPLAY_BUFFER_SIZE = config.REPLAY_BUFFER_SIZE

    # --- 초기화 ---
    env = PagePilotGNNAEnv(num_elements=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    node_feature_dim = env.observation_space['x'].shape[1]
    num_actions_per_node = env.action_space.nvec[1]

    policy_net = GNN_DQN(node_feature_dim, num_actions_per_node).to(device)
    target_net = GNN_DQN(node_feature_dim, num_actions_per_node).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
    steps_done = 0

    # --- 학습 루프 ---
    for i_episode in range(EPISODES):
        state_graph = env.reset()
        total_reward = 0

        for t in range(env.max_steps_per_episode):
            # --- 액션 선택 ---
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if random.random() > eps_threshold:
                with torch.no_grad():
                    state_graph = state_graph.to(device)
                    q_values = policy_net(state_graph)  # Shape: [노드 수, 행동 수]
                    # 가장 높은 Q-값을 가진 노드와 행동을 선택
                    action_node = q_values.argmax() // num_actions_per_node
                    action_move = q_values.argmax() % num_actions_per_node
                    action = (action_node.item(), action_move.item())
            else:
                action = env.action_space.sample()

            next_state_graph, reward, done, _ = env.step(action)
            total_reward += reward
            
            memory.push(state_graph, action, reward, next_state_graph, done)
            state_graph = next_state_graph

            # --- 모델 최적화 ---
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                
                # GNN에서는 배치를 순회하며 개별적으로 손실을 계산 (단순화된 접근)
                total_loss = 0
                for s, a, r, s_next, d in transitions:
                    s = s.to(device)
                    q_values = policy_net(s)
                    q_value_for_action = q_values[a[0], a[1]]

                    with torch.no_grad():
                        if d:
                            target_q_value = torch.tensor(r, device=device)
                        else:
                            s_next = s_next.to(device)
                            next_q_values = target_net(s_next)
                            max_next_q = next_q_values.max()
                            target_q_value = r + (GAMMA * max_next_q)
                    
                    total_loss += F.smooth_l1_loss(q_value_for_action.unsqueeze(0), target_q_value.unsqueeze(0))
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if done:
                break
        
        print(f"Episode {i_episode+1}, Total Reward: {total_reward:.4f}")

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    env.close()
    # 모델 저장
    os.makedirs(os.path.dirname(config.DQN_MODEL_PATH), exist_ok=True)
    torch.save(policy_net.state_dict(), config.DQN_MODEL_PATH)
    print(f"GNN 모델을 {config.DQN_MODEL_PATH}에 저장했습니다.")

if __name__ == "__main__":
    train_gnn_dqn()
