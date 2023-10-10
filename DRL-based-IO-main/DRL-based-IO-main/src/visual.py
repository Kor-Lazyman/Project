import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#df1=pd.read_csv("C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/DQN/DQN_csv.csv")
df2=pd.read_csv("C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/A2C/A2C_csv.csv")
df3=pd.read_csv("C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/SAC/SAC_csv.csv")
df4=pd.read_csv("C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/PPO/PPO_csv.csv")
plt.figure(figsize=(20, 10))
'''
plt.subplot(2, 2, 1)
plt.plot(df1,"g--",color='black')
plt.title("DQN")
'''
plt.plot(df3,label="SAC")
plt.plot(df4,label="PPO")
plt.plot(df2,label="A2C",color="green")

plt.xticks([0,500,1000],fontsize=40)
plt.yticks([-20000,-40000,-60000,-80000,-100000],fontsize=40)
plt.xlabel("Episodes",fontsize=40)
plt.ylabel("Total Reward",fontsize=40)
plt.legend(fontsize=40)
plt.show()