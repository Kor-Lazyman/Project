import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import Test_Data_Generate



class NonlinearModel(nn.Module):
    #ANN에 필요한 함수 설정
    def __init__(self):
        super(NonlinearModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
    
    
    #순전파 층 쌓기 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#변수 입력
num_cont = int(input("Enter the number of data points: "))
pred_cont =int(input("Enter the value for prediction: "))

#학습 데이터 X,y생성


#모델 선언
model = NonlinearModel()

#손실함수 설정
criterion = nn.MSELoss()

#최적화 함수 설정
optimizer = optim.Adam(model.parameters(), lr=0.01)

#학습 변수 설정
num_epoch = 1000

X=torch.linspace(0, num_cont,num_cont, dtype=torch.float32).reshape(-1, 1)
y=torch.Tensor(Test_Data_Generate.make_data(0,num_cont))

#학습 진행
for epoch in range(num_epoch):
    inputs = X
    targets = y

    # 순전파
    outputs=model(inputs)
    loss = criterion(outputs, targets)

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #100회마다 학습경과 출력
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
 
model.eval()        
X_test = torch.linspace(0, pred_cont, pred_cont, dtype=torch.float32).reshape(-1, 1)
y_test = torch.Tensor(Test_Data_Generate.make_data(0, pred_cont))

# 예측값 계산
with torch.no_grad():
    predictions = model(X_test)

rounded_predictions = torch.round(predictions)
# 평균 제곱 오차(MSE) 계산
mse = criterion(rounded_predictions, y_test) 
print(f'Mean Squared Error (MSE): {mse.item():.4f}')
