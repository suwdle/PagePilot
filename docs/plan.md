

# ✅ 전체 프로젝트 플랜 정리: RL 기반 UI 최적화

---

## 🎯 목표

> 실제 유저의 클릭/상호작용 데이터를 기반으로 UI 레이아웃을 최적화하는 강화학습 시스템을 구축한다. reward는 CTR 또는 dwell time 등의 사용자 반응으로 정의되며, agent는 UI 레이아웃을 조정해 높은 보상을 유도한다.

---

## 🔶 Phase 1. 데이터 수집 및 가공

### 1-1. UI Layout 데이터셋 확보 (state 구성)

| 데이터셋           | 용도                                        | 링크                                                                |
| -------------- | ----------------------------------------- | ----------------------------------------------------------------- |
| **WaveUI-25K** | 웹 기반 UI 구조 25K개 이상 (버튼, 텍스트 등 레이아웃 요소 포함) | [WaveUI (HF)](https://huggingface.co/datasets/Voxel51/WaveUI-25k) |
| **RICO**       | 모바일 앱 UI 구조 66K개 이상                       | [RICO](https://interactionmining.org/rico)                        |

📌 **목표**: UI layout을 상태(state) 벡터로 표현할 수 있도록 정규화된 벡터 생성

---

### 1-2. 유저 반응 로그 확보 (reward 예측용 학습 데이터)

| 데이터셋                        | 설명                            | CTR 포함 여부 |
| --------------------------- | ----------------------------- | --------- |
| **RetailRocket**            | 유저 세션별 click/view/purchase 로그 | ✅ CTR     |
| **Yahoo Front Page Logs**   | 뉴스 추천 클릭 로그                   | ✅ CTR     |
| **Avazu / Outbrain / MIND** | CTR 예측용 대규모 광고/뉴스 로그          | ✅ CTR     |

📌 **목표**: reward simulator 학습에 필요한 (UI feature → CTR) 데이터를 구성

---

## 🔶 Phase 2. 데이터 벡터화 및 전처리

### 2-1. UI Layout 벡터화 (state 표현)

* 방식 A: grid + channel 방식 (20×20×C tensor)
* 방식 B: 통계 기반 fixed vector (평균 위치, CTA 비율 등)
* 방식 C: GNN용 graph 구조 (선택적)

> 🔧 초기에는 방식 B를 추천 (속도 + 유연성)

### 2-2. CTR 라벨 구성

* RetailRocket의 클릭 여부 → binary label
* 레이아웃과 매칭된 click 여부 데이터를 만들어 `CTR 모델 학습용 샘플` 구성

---

## 🔶 Phase 3. Reward Simulator 학습

> UI 레이아웃 벡터를 입력으로 받아 CTR을 출력하는 모델

### 모델 후보

* `XGBoost`, `LightGBM`, `GradientBoostingClassifier`
* `MLPClassifier` (TensorFlow/Keras)
* `LogisticRegression` (baseline)

📌 **출력값**은 `CTR ∈ [0, 1]`이며, 강화학습에서 **보상 함수로 사용**됨

---

## 🔶 Phase 4. 강화학습 환경 구성

### 환경 요소 정의

| 구성요소           | 설명                              |
| -------------- | ------------------------------- |
| **state**      | 현재 UI 벡터 (또는 텐서)                |
| **action**     | UI 요소 조정 (예: 버튼 크기 증가, 위치 이동 등) |
| **reward**     | reward simulator가 예측한 CTR 값     |
| **transition** | action 반영 후 새로운 UI 상태 생성        |

📌 [OpenAI Gym 스타일 환경 클래스를 정의](https://www.gymlibrary.dev/)

---

## 🔶 Phase 5. RL 에이전트 학습

### 알고리즘 후보

* **DQN** (Deep Q-Network) ✅ 추천
* PPO / A2C (선택적)
* Policy Gradient (baseline)

> UI 조정 policy를 학습하여 CTR 최대화

---

## 🔶 Phase 6. 평가 및 시각화

| 평가 지표             | 설명                          |
| ----------------- | --------------------------- |
| 평균 CTR            | 테스트 세트에서 UI 최적화 후 예측 CTR    |
| A/B 테스트 시뮬        | 기존 vs 학습된 policy            |
| 행동 trajectory 시각화 | 에이전트가 UI를 어떻게 변화시키는지 로그/GIF |

---

# ✅ 요약된 전체 흐름도

```
UI Dataset ──▶ 벡터화 ──▶ CTR 예측 모델 (reward simulator)
                              │
        +─────────────────────┘
        │
      [Gym 환경 구성]  ◀─────── UI 변경 action
        │
     RL Agent 학습 (DQN)
        │
   최적 UI 레이아웃 도출
```

---

## 🔜 다음 단계 제안

1. WaveUI 데이터 샘플 기반 벡터화 코드 작성
2. RetailRocket을 기반으로 CTR 예측 모델 학습
3. Gym-style UI 환경 구성 시작
