import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import os

def train_new_simulator():
    """
    가상 데이터셋으로 새로운 보상 시뮬레이터(LightGBM)를 학습시킵니다.
    """
    print("--- 새로운 보상 시뮬레이터 학습 시작 ---")
    
    # 1. 데이터셋 로드
    dataset_path = 'data/synthetic_ui_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"에러: {dataset_path}에서 데이터셋을 찾을 수 없습니다.")
        print("`generate_dataset.py`를 먼저 실행해주세요.")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"{len(df)}개의 샘플로 데이터셋을 로드했습니다.")

    # 2. 피처(X)와 타겟(y) 정의
    features = [
        'pos_x', 'pos_y', 'size', 'contrast',
        'persona_power_user', 'persona_casual_browser', 'persona_elderly'
    ]
    target = 'click_probability'
    
    X = df[features]
    y = df[target]

    # 3. 학습용/테스트용 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"데이터를 {len(X_train)}개의 학습 샘플과 {len(X_test)}개의 테스트 샘플로 분리했습니다.")

    # 4. LightGBM 모델 초기화 및 학습
    print("LightGBM 모델을 학습합니다...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',  # Mean Absolute Error
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )

    # 5. 모델 평가
    print("\n모델 성능을 평가합니다...")
    y_pred = lgbm.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"테스트셋에 대한 모델의 R^2 점수: {r2:.4f}")

    # 6. 학습된 모델 저장
    model_path = 'models/reward_simulator_lgbm.joblib'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(lgbm, model_path)
    print(f"{model_path}에 학습된 모델을 성공적으로 저장했습니다.")

if __name__ == '__main__':
    train_new_simulator()
