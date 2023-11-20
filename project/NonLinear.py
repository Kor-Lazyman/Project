import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

num_cont = int(input("Enter the number of data points: "))
pred_cont = float(input("Enter the value for prediction: "))
X = torch.linspace(0, num_cont,num_cont, dtype=torch.float32).reshape(-1, 1)
y = 10 * torch.sin(X * 0.5 * np.pi)+random.randint(-1,1)
model = NonlinearModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epoch = 1000
for epoch in range(num_epoch):
    inputs = X
    targets = y

    # 순전파
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
        
model.eval()
x = torch.tensor([[pred_cont]], dtype=torch.float32)  # 예측을 위해 입력을 텐서로 변환
y_pred = model(x)

print("Pred:", y_pred.item())
