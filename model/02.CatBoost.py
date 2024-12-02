
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor


def create_many_to_one_detailed_output(data, target, input_window, output_window):

    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:i + input_window, :])  # 입력 시점의 모든 변수
        y.append(target[i + input_window : i + input_window + output_window])  # 출력 시점별 상세값
    return np.array(X), np.array(y)


# 입력 및 출력 윈도우 설정
input_window = 60  # 입력 시점 수
output_window = 7  # 출력 시점 수

df3 = df2.copy()
df3['주문일자'] = pd.to_datetime(df3['주문일자'])

# 연도별 분리
train_data = df3[(df3['주문일자'].dt.year == 2022) | (df3['주문일자'].dt.year == 2023)]
test_data = df3[df3['주문일자'].dt.year == 2024]


train_data.drop(columns='주문일자', inplace=True)
test_data.drop(columns='주문일자', inplace=True)

# 데이터셋 생성
X_train, y_train = create_many_to_one_detailed_output(
    train_data.values, train_data['총합계'].values, input_window, output_window
)
X_test, y_test = create_many_to_one_detailed_output(
    test_data.values, test_data['총합계'].values, input_window, output_window
)

# 데이터를 2D로 변환 (RandomForest에 맞게)
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (샘플 수, 입력 시점 * 변수 수)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 모델 학습

# 최적의 하이퍼파라미터 구성
best_params = {
    'early_stopping_rounds': 13,
    'learning_rate': 0.01900613042104467,
    'n_estimators': 1149
}

# CatBoostRegressor 모델 생성
best_model = CatBoostRegressor(
    early_stopping_rounds=best_params['early_stopping_rounds'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    verbose=0,# 학습 로그 출력을 원하지 않으면 0으로 설정
    random_state=42
)

model = MultiOutputRegressor(best_model)
model.fit(X_train_flat, y_train)

# 예측
y_pred = model.predict(X_test_flat)

# 평가
mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
print(f"Mean Absolute Error (MAE): {mae:.2f}")
