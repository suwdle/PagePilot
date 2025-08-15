# 작업 로그 (Work Log)

## 1. 프로젝트 분석 및 제안서 기반 개선 시작
- `propose.md`의 개선 계획을 바탕으로 작업 방향 수립.
- Playwright 기반 환경의 반복적인 실패로 인해, 안정적인 가상 환경에서의 모델 성능 검증으로 목표 전환.

## 2. 가상 데이터 생성 프레임워크 구축
- `enhanced_reward.py`: F-패턴 기반의 초기 CTR 생성기 구현.
- `synthetic_data_generator.py`: 사용자 페르소나(파워 유저, 일반 유저, 노년층)와 시선 패턴(F-패턴, Z-패턴)을 모델링하여, 더 현실적인 가상 클릭 확률을 생성하는 클래스 구현.
- `generate_dataset.py`: 위의 생성기를 사용하여, 새로운 보상 시뮬레이터 학습에 사용될 10,000개의 샘플을 포함한 `data/synthetic_ui_dataset.csv` 데이터셋 생성.

## 3. 신규 보상 시뮬레이터 학습
- `train_reward_simulator.py`: 생성된 가상 데이터셋을 사용하여 LightGBM 기반의 보상 예측 모델을 학습.
- 학습된 모델은 R^2 점수 0.9175를 기록하며, `models/reward_simulator_lgbm.joblib`로 저장됨.

## 4. 강화학습 환경 업데이트
- `rl_env.py`: 학습된 LightGBM 모델(`reward_simulator_lgbm.joblib`)을 사용하도록 보상 계산 로직 수정.

## 5. 성능 시각화 대시보드 구축
- `dashboard.py`: Streamlit을 사용하여 학습된 에이전트의 최적화 과정을 실시간으로 시각화하는 대시보드 구현.
- 다크 테마, `st.metric`, `st.expander` 등을 사용하여 UI/UX 개선.

## 6. GNN 기반 환경으로 확장
- `rl_env.py`: 단일 요소 환경에서 벗어나, 여러 UI 요소를 그래프(노드, 엣지)로 표현하는 다중 요소 환경으로 재구축.
- `gnn_agent.py`: 그래프 형태의 상태를 처리할 수 있는 GNN 기반 DQN 모델 정의.
- `gnn_trainer.py`: 새로운 GNN 환경과 모델을 사용하여 학습을 수행하는 트레이너 스크립트 작성.

## 7. GNN 대시보드 업데이트
- `dashboard.py`: GNN 환경과 모델에 맞게 대시보드 로직 및 시각화 방법 업데이트.
