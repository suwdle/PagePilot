import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from rl_env import PagePilotEnv
from dqn_trainer import DQN

def visualize_results():
    """학습된 에이전트의 성능을 시각화합니다."""
    print("--- 학습 결과 시각화 시작 ---")

    # 1. 환경 및 모델 로드
    env = PagePilotEnv(persona='casual_browser')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy_net = DQN(input_dim, output_dim).to(device)
    policy_net.load_state_dict(torch.load(config.DQN_MODEL_PATH))
    policy_net.eval() # 평가 모드로 설정

    print("환경 및 학습된 모델을 로드했습니다.")

    # 2. 평가 실행
    num_eval_episodes = 50
    episode_rewards = []
    print(f"{num_eval_episodes}개의 에피소드에 대해 평가를 실행합니다...")

    for i_episode in range(num_eval_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                # Epsilon-greedy가 아닌, 학습된 정책을 직접 사용
                action = policy_net(state_tensor).max(1)[1].item()
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"에피소드 {i_episode + 1}: 총 보상 = {total_reward:.2f}")

    env.close()
    print("평가 실행 완료.")

    # 3. 학습 곡선 시각화
    print("학습 곡선을 생성합니다...")
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_eval_episodes + 1), episode_rewards, marker='o', linestyle='-')
    plt.title('Agent Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # 결과 저장
    output_path = os.path.join(config.ROOT_DIR, 'results', 'learning_curve.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    
    print(f"학습 곡선 그래프를 {output_path}에 저장했습니다.")
    print("--- 시각화 완료 ---")

if __name__ == "__main__":
    visualize_results()
