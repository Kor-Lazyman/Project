import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/A2C/A2C_csv.csv")
plt.plot(df)
plt.show()