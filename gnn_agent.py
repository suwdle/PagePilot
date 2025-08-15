import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN_DQN(nn.Module):
    """
    그래프 신경망(GNN) 기반 DQN 에이전트입니다.
    UI 상태를 그래프로 입력받아, 각 UI 요소(노드)에 대한 행동의 Q-값을 출력합니다.
    """
    def __init__(self, node_feature_dim, num_actions_per_node, hidden_dim=128):
        """
        Args:
            node_feature_dim (int): 각 노드의 특징 벡터 차원 (예: 5개 - x,y,w,h,type)
            num_actions_per_node (int): 각 노드에 대해 수행할 수 있는 행동의 수 (예: 8개)
            hidden_dim (int): 은닉층의 차원
        """
        super(GNN_DQN, self).__init__()
        
        # 그래프 컨볼루션 레이어
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 각 노드에 대한 행동의 가치를 계산하는 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions_per_node)
        )

    def forward(self, data):
        """
        GNN 모델의 순전파 과정입니다.
        
        Args:
            data (torch_geometric.data.Data): 그래프 데이터 객체
                - x (Tensor): 노드 특징 행렬 [노드 수, 노드 특징 차원]
                - edge_index (LongTensor): 그래프 연결성 정보 [2, 엣지 수]
        
        Returns:
            Tensor: 각 노드에 대한 각 행동의 Q-값 [노드 수, 노드당 행동 수]
        """
        x, edge_index = data.x, data.edge_index
        
        # 그래프 컨볼루션 적용
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x) # 결과 shape: [노드 수, 은닉층 차원]
        
        # 각 노드의 임베딩을 사용하여 행동의 가치를 계산
        q_values_per_node = self.action_head(x)
        
        return q_values_per_node

if __name__ == '__main__':
    # --- GNN_DQN 모델 테스트 ---
    from torch_geometric.data import Data

    print("--- GNN_DQN 모델 테스트 ---")
    
    num_nodes = 5
    node_feature_dim = 5
    num_actions = 8
    
    x = torch.randn(num_nodes, node_feature_dim)
    edge_list = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    sample_graph_data = Data(x=x, edge_index=edge_index)
    
    model = GNN_DQN(node_feature_dim=node_feature_dim, num_actions_per_node=num_actions)
    print("모델 초기화 완료:")
    print(model)
    
    q_values = model(sample_graph_data)
    
    print(f"\n입력 그래프: {sample_graph_data}")
    print(f"출력 Q-값 텐서 shape: {q_values.shape}")
    print("(예상 shape: [노드 수, 노드당 행동 수])")
    print("출력 Q-값:")
    print(q_values)
