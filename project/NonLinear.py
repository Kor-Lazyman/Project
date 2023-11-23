import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

class NonlinearModel(nn.Module):
    def __init__(self):
        super(NonlinearModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 데이터 로드
df = pd.read_excel("C:/Users/User/Desktop/Non-Linear/Data.xlsx")

#시계열 데이터로 변환
df['DUEDATE'] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d')
df['DUEDATE'] = (df['DUEDATE'] - pd.to_datetime('1970-01-01')).dt.total_seconds() / (24 * 60 * 60)

#OutLier 제거
df = df[df['DEMAND_QTY'] < 10000]

# PyTorch 텐서로 변환
X_train = torch.from_numpy(np.array(df.iloc[:1000, 0], dtype=np.float32)).view(-1, 1)  
y_train = torch.from_numpy(np.array(df.iloc[:1000,1], dtype=np.float32))  



# 모델, 손실 함수, 최적화 함수 설정
model = NonlinearModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 진행
num_epoch = 10000
for epoch in range(num_epoch):
    inputs = X_train
    targets = y_train

    # 순전파
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100회마다 학습경과 출력
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')

# 학습된 모델을 평가
model.eval()

# 테스트 데이터 생성
X_test = torch.from_numpy(np.array(df.iloc[1000:, 0], dtype=np.float32)).view(-1, 1)  
y_test = torch.from_numpy(np.array(df.iloc[1000:, 1], dtype=np.float32))  

# 예측값 계산
with torch.no_grad():
    predictions = model(X_test)

# MSE 계산
mse = criterion(predictions, y_test)
print(f'Mean Squared Error (MSE): {mse.item():.4f}')

X_test = torch.from_numpy(np.array(df.iloc[0, 0], dtype=np.float32)).view(-1, 1)

# 예측값 계산
with torch.no_grad():
    predictions = model(X_test)
rounded_predictions = torch.round(predictions)
# 결과 출력
print("모델이 예측한 값:", rounded_predictions.item())