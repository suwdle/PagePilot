import numpy as np

class UserPersona:
    """특정 행동 패턴을 가진 사용자 유형을 정의합니다."""
    def __init__(self, name, scan_pattern, click_threshold, sensitivity_factors=None):
        self.name = name
        self.scan_pattern = scan_pattern
        self.click_threshold = click_threshold
        self.sensitivity = sensitivity_factors if sensitivity_factors else {}

class SyntheticDataGenerator:
    """
    UI 레이아웃과 사용자 페르소나를 기반으로 가상의 사용자 인터랙션 데이터를 생성합니다.
    """
    def __init__(self, grid_size=(20, 20)):
        self.grid_size = grid_size
        self.personas = self._load_personas()
        self.heatmaps = {
            'f_pattern': self._create_f_pattern_heatmap(),
            'z_pattern': self._create_z_pattern_heatmap()
        }

    def _load_personas(self):
        """사전에 정의된 사용자 페르소나를 로드합니다."""
        personas_data = {
            "power_user": {
                "scan_pattern": "f_pattern",
                "click_threshold": 0.3, # 클릭에 덜 민감 (쉽게 클릭)
                "sensitivity_factors": {"size": 1.0, "contrast": 1.0}
            },
            "casual_browser": {
                "scan_pattern": "z_pattern",
                "click_threshold": 0.7, # 클릭에 더 민감 (잘 클릭 안함)
                "sensitivity_factors": {"size": 1.2, "contrast": 0.8}
            },
            "elderly": {
                "scan_pattern": "f_pattern",
                "click_threshold": 0.5,
                "sensitivity_factors": {"size": 1.5, "contrast": 1.2} # 크기와 대비에 더 민감
            }
        }
        return {name: UserPersona(name, **data) for name, data in personas_data.items()}

    def _create_f_pattern_heatmap(self):
        """F-형태 스캔 패턴을 기반으로 히트맵을 생성합니다."""
        rows, cols = self.grid_size
        heatmap = np.zeros((rows, cols))
        heatmap[:int(rows * 0.2), :] += 0.8
        heatmap[int(rows * 0.4):int(rows * 0.6), :int(cols * 0.5)] += 0.5
        heatmap[:, :int(cols * 0.2)] += 0.6
        max_val = np.max(heatmap)
        return heatmap / max_val if max_val > 0 else heatmap

    def _create_z_pattern_heatmap(self):
        """Z-형태 스캔 패턴을 기반으로 히트맵을 생성합니다."""
        rows, cols = self.grid_size
        heatmap = np.zeros((rows, cols))
        heatmap[0, :] = 1.0
        for i in range(min(rows, cols)):
            heatmap[i, cols - 1 - i] = 0.8
        heatmap[rows - 1, :] = 1.0
        max_val = np.max(heatmap)
        return heatmap / max_val if max_val > 0 else heatmap

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_click_probability(self, ui_element, persona_name='casual_browser'):
        """
        주어진 페르소나에 대해 단일 UI 요소의 클릭 확률을 계산합니다.
        """
        persona = self.personas.get(persona_name)
        if not persona:
            raise ValueError(f"페르소나 '{persona_name}'를 찾을 수 없습니다.")

        heatmap = self.heatmaps.get(persona.scan_pattern, self.heatmaps['f_pattern'])
        x, y = ui_element.get('position', (0.5, 0.5))
        row = np.clip(int(y * (self.grid_size[0] - 1)), 0, self.grid_size[0] - 1)
        col = np.clip(int(x * (self.grid_size[1] - 1)), 0, self.grid_size[1] - 1)
        visual_weight = heatmap[row, col]

        size_factor = ui_element.get('size', 1.0) * persona.sensitivity.get('size', 1.0)
        contrast_factor = ui_element.get('contrast', 1.0) * persona.sensitivity.get('contrast', 1.0)
        
        logit = visual_weight + (0.5 * size_factor) + (0.3 * contrast_factor) - persona.click_threshold
        logit += np.random.normal(0, 0.1)

        return self._sigmoid(logit)

if __name__ == '__main__':
    generator = SyntheticDataGenerator()
    
    sample_element = {
        'position': (0.15, 0.2), # F-패턴에서 점수가 높은 좌측 상단
        'size': 1.2, # 평균보다 20% 큼
        'contrast': 1.5 # 평균보다 50% 높은 대비
    }

    print("--- 가상 데이터 생성 테스트 ---")
    
    prob_power_user = generator.calculate_click_probability(sample_element, 'power_user')
    print(f"파워 유저의 클릭 확률: {prob_power_user:.4f}")

    prob_casual = generator.calculate_click_probability(sample_element, 'casual_browser')
    print(f"일반 유저의 클릭 확률: {prob_casual:.4f}")
    
    prob_elderly = generator.calculate_click_probability(sample_element, 'elderly')
    print(f"노년층 유저의 클릭 확률: {prob_elderly:.4f}")
