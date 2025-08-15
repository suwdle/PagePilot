import numpy as np

class EnhancedCTRGenerator:
    """
    UI 레이아웃에 대해 더 현실적인 클릭률(CTR)을 생성하는 클래스입니다.
    이 생성기는 다음을 고려합니다:
    1. 시각적 스캔을 위한 F-패턴 가중치.
    2. UI 요소 수에 기반한 인지 부하 (밀러의 법칙).
    3. 현실적인 변동성을 시뮬레이션하기 위한 노이즈.
    """
    def __init__(self, grid_size=(20, 20)):
        """
        생성기를 초기화하고 F-패턴 히트맵을 생성합니다.
        """
        self.grid_size = grid_size
        self.heatmap = self._create_f_pattern_heatmap()

    def _create_f_pattern_heatmap(self):
        """
        F-형태 스캔 패턴을 기반으로 히트맵을 생성합니다.
        좌측 상단 영역이 더 높은 가중치를 갖습니다.
        """
        rows, cols = self.grid_size
        heatmap = np.zeros((rows, cols))

        # 'F'의 상단 가로선
        heatmap[:int(rows * 0.2), :] += 0.8
        
        # 'F'의 중간 가로선
        heatmap[int(rows * 0.4):int(rows * 0.6), :int(cols * 0.5)] += 0.5

        # 'F'의 수직 기둥
        heatmap[:, :int(cols * 0.2)] += 0.6

        # 최대값이 1.0이 되도록 정규화
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        
        return np.clip(heatmap, 0, 1)

    def _calculate_cognitive_load_penalty(self, num_elements):
        """
        밀러의 법칙(7±2)에 영감을 받아 요소 수에 따른 페널티를 계산합니다.
        요소 수가 특정 임계값(예: 9)을 초과하면 페널티가 적용됩니다.
        """
        # 최적의 요소 수를 7개로 가정
        optimal_elements = 7
        # 9개를 초과하는 요소부터 페널티를 점진적으로 부과
        penalty = max(0, (num_elements - (optimal_elements + 2)) * 0.05)
        return penalty

    def generate_realistic_ctr(self, state):
        """
        주어진 UI 상태(state)에 대해 현실적인 CTR을 생성합니다.

        Args:
            state (np.ndarray): UI의 현재 상태를 나타내는 numpy 배열.
                                  [pos_x, pos_y, num_elements, ...] 형태를 가정합니다.
        
        Returns:
            float: 시뮬레이션된 CTR 값.
        """
        if not isinstance(state, np.ndarray) or state.shape[0] < 3:
            # 상태 벡터가 예상과 다를 경우 기본값 반환
            return 0.0

        pos_x, pos_y, num_elements = state[0], state[1], state[2]

        # 위치를 grid_size에 맞게 스케일링
        row = int(pos_y * (self.grid_size[0] - 1))
        col = int(pos_x * (self.grid_size[1] - 1))
        
        # 좌표가 범위를 벗어나지 않도록 클리핑
        row = np.clip(row, 0, self.grid_size[0] - 1)
        col = np.clip(col, 0, self.grid_size[1] - 1)

        # 1. F-패턴 히트맵에서 기본 점수 가져오기
        base_score = self.heatmap[row, col]
        
        # 2. 인지 부하 페널티 계산
        cognitive_load_penalty = self._calculate_cognitive_load_penalty(num_elements)
        
        # 3. 현실적인 노이즈 추가
        noise = np.random.normal(0, 0.05)
        
        # 최종 CTR 계산 및 0과 1 사이로 클리핑
        final_ctr = base_score - cognitive_load_penalty + noise
        return float(np.clip(final_ctr, 0, 1))

