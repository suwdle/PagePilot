import numpy as np
import pandas as pd
from synthetic_data_generator import SyntheticDataGenerator
import random
import os

def generate_dataset(num_samples=10000):
    """
    보상 시뮬레이터 모델 학습을 위한 합성 데이터셋을 생성합니다.
    """
    print(f"{num_samples}개의 가상 데이터 생성을 시작합니다...")
    
    generator = SyntheticDataGenerator()
    personas = list(generator.personas.keys())
    
    data = []
    
    for i in range(num_samples):
        # 1. UI 요소 특징을 무작위로 생성
        element_features = {
            'position': (random.uniform(0, 1), random.uniform(0, 1)),
            'size': random.uniform(0.5, 2.0),  # 평균 크기의 0.5배 ~ 2배
            'contrast': random.uniform(0.5, 2.0)  # 평균 대비의 0.5배 ~ 2배
        }
        
        # 2. 페르소나를 무작위로 선택
        persona_name = random.choice(personas)
        
        # 3. 클릭 확률 계산 (이것이 정답 라벨이 됨)
        click_probability = generator.calculate_click_probability(element_features, persona_name)
        
        # 4. 모델이 학습할 수 있도록 데이터를 펼쳐서 저장
        record = {
            'pos_x': element_features['position'][0],
            'pos_y': element_features['position'][1],
            'size': element_features['size'],
            'contrast': element_features['contrast'],
            'persona_power_user': 1 if persona_name == 'power_user' else 0,
            'persona_casual_browser': 1 if persona_name == 'casual_browser' else 0,
            'persona_elderly': 1 if persona_name == 'elderly' else 0,
            'click_probability': click_probability
        }
        data.append(record)

    print("데이터셋 생성 완료.")
    return pd.DataFrame(data)

if __name__ == '__main__':
    # 데이터셋 생성
    synthetic_dataset = generate_dataset(num_samples=10000)
    
    # 저장 경로 설정
    output_path = 'data/synthetic_ui_dataset.csv'
    
    # `data` 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 데이터셋을 CSV 파일로 저장
    synthetic_dataset.to_csv(output_path, index=False)
    
    print(f"데이터셋을 {output_path}에 성공적으로 저장했습니다.")
    print("--- 데이터셋 미리보기 ---")
    print(synthetic_dataset.head())
