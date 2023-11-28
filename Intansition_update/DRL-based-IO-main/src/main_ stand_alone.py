from InventoryMgtEnv import GymInterface
from config import *
import numpy as np
import optuna
import optuna.visualization as vis
import environment as env
from stable_baselines3.common.evaluation import evaluate_policy
import time


# Create environment
simpy_env, inventoryList, procurementList,productionList,sales, customer, providerList, daily_events = env.create_env(
     I, P, DAILY_EVENTS)
env.simpy_event_processes(simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events, I)
total_reward = 0
num_episode = 1 
for x in range(SIM_TIME):
    simpy_env.run(until=simpy_env.now+24)
    daily_total_cost = env.cal_daily_cost_ACC(
        inventoryList, procurementList, productionList, sales)
    for y in range(1,len(I)):
        I[y]['LOT_SIZE_ORDER']=random.randint(0,5)
    if PRINT_SIM_EVENTS:
        # Print the simulation log every 24 hours (1 day)
        print(f"\nDay {(simpy_env.now+1) // 24}:")
        for y in range(1,len(I)):
            print(f"[Order Quantity for {I[y]['NAME']}]  {I[y]['LOT_SIZE_ORDER']}")
    
        for log in daily_events:
            print(log)
        print("[Daily Total Cost] ", daily_total_cost)
    daily_events.clear()
    reward = -daily_total_cost
    total_reward += reward
print(total_reward)



'''
#모델 저장 및 로드 (선택적)
model.save("dqn_inventory")
loaded_model = DQN.load("dqn_inventory")
'''

# TensorBoard 실행:
# tensorboard --logdir="C:/tensorboard_logs/"
# http://localhost:6006/
