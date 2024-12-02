
from joblib import load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
from datetime import datetime, timedelta


## model1 : multiregression

# 파일 경로
file_path = '/content/catboost_multioutput_model.pkl'

# 모델 불러오기
model1 = load(file_path)

## 날짜 설정
start_date = datetime(2024, 3, 31) - timedelta(days=59)
end_data = datetime(2024, 3, 31)

target_data = df2[(df2['주문일자'] >= start_date) & (df2['주문일자']<=end_data)]
target_data2 = target_data.drop(columns='주문일자')
input_array = np.array(target_data2).flatten()

model1_yhat = model1.predict(input_array)

mode1_y = df2[(df2['주문일자'] >= datetime(2024,4,1)) & (df2['주문일자'] < datetime(2024,4,1) + timedelta(days=7))]['총합계']

mae = mean_absolute_error(mode1_y, model1_yhat)

## lstm 모델 불러오기
checkpoint = torch.load('/content/lstm_model_with_scalers.pth')

# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, output_size, drop_out,layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.drop_out = drop_out
        self.layers = layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_size, output_size,bias=True)
    def forward(self, x):

        x, _hidden_state = self.lstm(x)
        x = self.fc(x[:, -1, :])  # 마지막 타임스텝만 사용
        return x

torch.manual_seed(0)  ## seed 고정
seq_length = 90        ## 지난 90일 데이터를 통해 예측
input_size = 19        ## input 사이즈 (변수 수)
hidden_size = 64     ## hidden state 사이즈
output_size = 30      ## output 사이즈 (미래 30일 예측)
layers=4
drop_out=0.4
model = LSTM(input_size, hidden_size,seq_length,output_size, drop_out, layers)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
## 날짜 설정
lstm_start_date = datetime(2024, 3, 31) - timedelta(days=89)
lstm_end_data = datetime(2024, 3, 31)


target_data = model_data[(model_data['주문일자'] >= lstm_start_date) & (model_data['주문일자']<= lstm_end_data)].drop(columns='주문일자')
target_data['holiday'] = target_data['holiday'].astype(int)
target_data = checkpoint['scaler_x'].transform(target_data.values)
lstm_input = torch.FloatTensor(np.array(target_data))
lstm_input = lstm_input.unsqueeze(0)
lstm_yhat = model(lstm_input)
lstm_yhat = checkpoint['scaler_y'].inverse_transform(lstm_yhat.detach().numpy()).flatten()

lstm_y = df2[(df2['주문일자'] >= datetime(2024,4,1)) & (df2['주문일자'] < datetime(2024,4,1) + timedelta(days=30))]['총합계']

lstm_mae = mean_absolute_error(lstm_y, lstm_yhat)


fin = pd.DataFrame({'y' : lstm_y,
                    'lstm':lstm_yhat}).reset_index()
model1_yhat_df= pd.DataFrame(model1_yhat, columns=['model1'])

fin = pd.concat([fin,model1_yhat_df], axis=1)
fin['model1'] = fin['model1'].fillna(fin['lstm'])

fin_mae = mean_absolute_error(fin['y'], fin['model1'])
fin['fin'] = fin['lstm']*0.66 + fin['model1']*0.34   ## 비율 설정
fin_mae = mean_absolute_error(fin['y'], fin['fin'])

print('reg_model:{} , lstm_mode :{}, ensembel_model: {}'.format(mae,lstm_mae,fin_mae))
