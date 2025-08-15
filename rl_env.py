import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import joblib
import os
import config

class PagePilotGNNAEnv(gym.Env):
    """
    여러 UI 요소를 그래프 형태로 표현하는 GNN 기반 환경입니다.
    - 상태(State): 노드(UI 요소)와 엣지(요소 간 관계)로 구성된 그래프 데이터
    - 액션(Action): (조작할 요소 ID, 행동 ID)의 조합
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_elements=5, persona='casual_browser'):
        super().__init__()
        self.num_elements = num_elements
        self.persona = persona
        
        # 액션: N개의 요소 중 하나를 선택하고, 8가지 행동 중 하나를 수행
        self.action_space = spaces.MultiDiscrete([self.num_elements, 8])
        
        # 관찰 공간: 그래프 (Gym 호환을 위해 Dict 형태로 정의)
        # 노드 특징: [x, y, 너비, 높이, 요소 타입]
        node_feature_dim = 5 
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0, high=1, shape=(self.num_elements, node_feature_dim), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.num_elements-1, shape=(2, self.num_elements * (self.num_elements - 1)), dtype=np.int64)
        })

        model_path = 'models/reward_simulator_lgbm.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"보상 시뮬레이터 모델을 찾을 수 없습니다: {model_path}. 먼저 train_reward_simulator.py를 실행하세요.")
        self.reward_simulator = joblib.load(model_path)

        self.current_step = 0
        self.max_steps_per_episode = config.MAX_STEPS_PER_EPISODE
        self.graph_state = None
        self.previous_potential = 0

    def _get_potential(self, graph_state):
        """UI 그래프 전체의 잠재적 가치(보상)를 계산합니다."""
        total_potential = 0
        nodes = graph_state.x.cpu().numpy() # Convert to numpy for processing
        
        # 각 노드(UI 요소)의 잠재적 가치를 합산
        for node in nodes:
            x, y, width, height, elem_type = node
            
            features = {
                'pos_x': x, 'pos_y': y, 'size': width * height, 'contrast': 1.0,
                'persona_power_user': 1 if self.persona == 'power_user' else 0,
                'persona_casual_browser': 1 if self.persona == 'casual_browser' else 0,
                'persona_elderly': 1 if self.persona == 'elderly' else 0,
            }
            feature_df = pd.DataFrame([features])
            training_columns = ['pos_x', 'pos_y', 'size', 'contrast', 'persona_power_user', 'persona_casual_browser', 'persona_elderly']
            feature_df = feature_df[training_columns]
            
            total_potential += self.reward_simulator.predict(feature_df)[0]
        
        return total_potential

    def reset(self):
        """환경을 여러 개의 무작위 UI 요소를 가진 상태로 초기화합니다."""
        self.current_step = 0
        
        node_features = []
        for i in range(self.num_elements):
            elem_type = 0 if i == 0 else 1 # 0: 버튼, 1: 텍스트
            node_features.append([
                np.random.uniform(0.1, 0.8), # x
                np.random.uniform(0.1, 0.8), # y
                np.random.uniform(0.1, 0.2), # width
                np.random.uniform(0.05, 0.1), # height
                elem_type
            ])
        x = torch.tensor(node_features, dtype=torch.float32)

        # 모든 노드가 서로 연결된 완전 그래프(fully connected graph) 생성
        edge_list = []
        for i in range(self.num_elements):
            for j in range(self.num_elements):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        self.graph_state = Data(x=x, edge_index=edge_index)
        self.previous_potential = self._get_potential(self.graph_state)
        
        return self.graph_state

    def step(self, action):
        element_id, move_id = action
        self.current_step += 1

        # 1. Get a mutable copy of node features as a standard list of floats
        node_features_tensor = self.graph_state.x.clone()
        selected_node_tensor = node_features_tensor[element_id]
        
        # 2. Convert tensor to simple Python floats for manipulation
        x, y, w, h, elem_type = [v.item() for v in selected_node_tensor]

        # 3. Apply actions to the Python floats
        move_amount = 0.02
        size_amount = 0.01

        if move_id == 0: y -= move_amount
        elif move_id == 1: y += move_amount
        elif move_id == 2: x -= move_amount
        elif move_id == 3: x += move_amount
        elif move_id == 4: w += size_amount
        elif move_id == 5: w = max(0.05, w - size_amount)
        elif move_id == 6: h += size_amount
        elif move_id == 7: h = max(0.05, h - size_amount)
        
        # 4. Update the original tensor with the modified floats
        node_features_tensor[element_id] = torch.tensor([x, y, w, h, elem_type])
        self.graph_state.x = node_features_tensor

        # 5. Check for boundary violations with the clear float values
        is_out_of_bounds = (x < 0 or y < 0 or (x + w) > 1 or (y + h) > 1)
        
        if is_out_of_bounds:
            reward = -1.0
            done = True
        else:
            new_potential = self._get_potential(self.graph_state)
            reward = new_potential - self.previous_potential
            self.previous_potential = new_potential
            done = self.current_step >= self.max_steps_per_episode

        return self.graph_state, reward, done, {}

    def close(self):
        pass