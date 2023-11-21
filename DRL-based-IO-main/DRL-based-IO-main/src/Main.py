# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:01:05 2023

@author: User
"""
import csv
from argparse import Action
from calendar import c
from pyexpat import model
from re import T
from stringprep import c22_specials
from tkinter import SEL
import environment as Env
import numpy as np
import random
import time 
from config import *
import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import SAC
import gym
from gym import spaces
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import os

# TensorBoard 로깅을 위한 디렉토리 경로 설정
log_dir=f"./logs/{RL_ALGORITHM}"

class CustomSimPyEnv(gym.Env):
    def __init__(self, daily_events,action_space):
        self.total_reward_list=[]
        self.action_space2=action_space
        self.total_reward=0
        self.due_date=0
        self.episode=0
        self.day=1
        self.cont=0
        self.daily_events = daily_events
        super(CustomSimPyEnv, self).__init__()
        self.time = 0
        if RL_ALGORITHM=="PPO" or RL_ALGORITHM=="SAC":
            self.action_space=spaces.Box(low=0, high=4, shape=(3,), dtype=np.float32)
        elif RL_ALGORITHM=="A2C":
            self.action_space=spaces.MultiDiscrete([5,5,5])
        self.observation_space = spaces.Dict({
            'Day':spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'DueDate':spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'raw_material_inventory 1': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'raw_material_inventory 2': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'raw_material_inventory 3': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'WIP':spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'product_inventory': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'current_demand': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
        })
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = Env.create_env(I, P, daily_events)
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.current_observation = {
            'Day':np.array([self.day]),
            'DueDate':np.array([self.due_date]),
            'raw_material_inventory 1': np.array([I[1]["INIT_LEVEL"]]),
            'raw_material_inventory 2': np.array([I[2]["INIT_LEVEL"]]),
            'raw_material_inventory 3': np.array([I[3]["INIT_LEVEL"]]),
            'WIP':np.array([I[4]["INIT_LEVEL"]]),
            'product_inventory': np.array([self.inventoryList[0].current_level]),
            'current_demand': np.array([0]),

        }
        
        self.episode_length = 1

    def reset(self):
       
        self.check=False
        self.day=1
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.due_date=0
        daily_events = []
        self.total_reward=0
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = Env.create_env(I, P, daily_events)
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.current_observation = {
            'Day':np.array([self.day]),
            'DueDate':np.array([self.due_date]),
            'raw_material_inventory 1': np.array([I[1]["INIT_LEVEL"]]),
            'raw_material_inventory 2': np.array([I[2]["INIT_LEVEL"]]),
            'raw_material_inventory 3': np.array([I[3]["INIT_LEVEL"]]),
            'WIP':np.array([I[4]["INIT_LEVEL"]]),
            'product_inventory': np.array([I[0]["INIT_LEVEL"]]),
            'current_demand': np.array([0]),
        }

        return self.current_observation

    def step(self, action):
        reward=0
        daily_total_cost = 0
        self.simpy_env.run(until=self.simpy_env.now+24)      
        if self.day%I[0]["CUST_ORDER_CYCLE"]==1:
            self.due_date=I[0]['DUE_DATE']
        
        else:
            self.due_date= self.due_date-1
            
        if self.simpy_env.now%24==0:
                print(f"\nDay {self.day}:")
                for x in range(len(self.procurementList)):  
                    print("orders:",action[x],"Time",self.simpy_env.now)
                     # Procurements
                    self.simpy_env.process(self.procurementList[x].order(
                    self.providerList[x], self.inventoryList[self.providerList[x].item_id], self.daily_events,self.simpy_env.now,action[x]))

                for inven in self.inventoryList:
                    daily_total_cost += inven.daily_inven_cost
                    print("id:", inven.item_id, "inven:", inven.current_level, "Cost:", inven.daily_inven_cost)
                    inven.daily_inven_cost = 0
                for production in self.productionList:
                    daily_total_cost += production.daily_production_cost
                    print("Production_cost",production.daily_production_cost)
                    production.daily_production_cost = 0
                for procurement in self.procurementList:
                    daily_total_cost += procurement.daily_procurement_cost
                    print("id:",procurement.item_id,"cost",procurement.daily_procurement_cost)
                    procurement.daily_procurement_cost = 0
                daily_total_cost += self.sales.daily_selling_cost
                print("sales cost:", self.sales.daily_selling_cost)
                print("shortage:",self.sales.num_shortages)
                self.sales.daily_selling_cost=0
                reward =-daily_total_cost
                print("[Daily Total reward] ", reward)
                self.total_reward=self.total_reward+reward
                
        self.current_observation = {
                'Day':np.array([self.day]),
                'DueDate':np.array([self.due_date]),
                'raw_material_inventory 1': np.array([self.inventoryList[1].current_level]),
                'raw_material_inventory 2': np.array([self.inventoryList[2].current_level]),
                'raw_material_inventory 3': np.array([self.inventoryList[3].current_level]),
                'WIP':np.array([self.inventoryList[4].current_level]),
                'product_inventory': np.array([I[0]["INIT_LEVEL"]]),
                
                'current_demand': np.array([self.customer.order_history[-1]]),
            }  
        done = self.simpy_env.now >= 100* 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            print("Total reward: ", self.total_reward)
            self.total_reward_list.append(self.total_reward)
            self.total_reward = 0
            self.episode += 1

        self.day=self.day+1
        print(self.episode)

        return self.current_observation, reward, done, {}

        

    
def main():
    daily_events = []
   
    env = CustomSimPyEnv(daily_events,action_space)
    # 에피소드 수를 조정
    # model을 불러올려면 model=(A2C).load->
    if RL_ALGORITHM=="SAC":
        model1 = SAC('MultiInputPolicy', env, verbose=1, )
    elif RL_ALGORITHM=="A2C":
        model1 = A2C('MultiInputPolicy', env, verbose=1,n_steps=100)
    elif RL_ALGORITHM=="PPO":
        model1 = PPO('MultiInputPolicy', env, verbose=1,n_steps=100 )
 
   
   # model=[model1,model2,model3]
    model1.learn(total_timesteps=100*10000)
    history=env.total_reward_list
    print(len(history))
            #재고제한 변경
    os.chdir('C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/SAC')
    history=pd.DataFrame(history)
    history.to_csv('SAC_csv.csv')
    model1.save("SAC_MODEL")
    
if __name__ == "__main__":
    main()