# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:01:05 2023

@author: User
"""

from argparse import Action
from calendar import c
from pyexpat import model
from re import T
from stringprep import c22_specials
from tkinter import SEL
import environment as Env
import numpy as np
import random
from visualization import *
import time 
from config import *
import gym
from stable_baselines3 import A2C
import gym
from gym import spaces
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import os

# TensorBoard 로깅을 위한 디렉토리 경로 설정
log_dir="./logs/A2C"
    
class CustomSimPyEnv(gym.Env):
    def __init__(self, daily_events,action_space):
        self.total_reward_list=[]
        self.action_space2=action_space
        self.total_reward=0
        self.due_date=0
        self.episode=0
        self.day=0
        self.cont=0
        self.daily_events = daily_events
        super(CustomSimPyEnv, self).__init__()
        self.time = 0
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
        self.total_reward_list.append(self.total_reward)
        self.check=False
        self.day=0
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
            'product_inventory': np.array([self.inventoryList[0].current_level]),
            'current_demand': np.array([0]),
        }
        if self.episode==10000:
            self.writer.close()
        return self.current_observation

    def step(self, action):
        reward=0
        daily_total_cost = 0
        if (self.simpy_env.now) % 24 == 0:
            
            print(self.episode)
            self.day=self.day+1
         
        if self.day>SIM_TIME:
            self.episode+=1
            self.obs=self.reset()
            if self.day%I[0]["CUST_ORDER_CYCLE"]==1:
                self.due_date=I[0]['DUE_DATE']
                if self.due_date==0:
                    pass
                else:
                    self.due_date= self.due_date-1

              
            if self.simpy_env.now%24==0 :
                self.day=self.day+1
                print(f"\nDay {self.day}:")
                for x in range(len(self.procurementList)):  
                    print("orders:",action[x])
                    self.simpy_env.process(self.procurementList[x].order(
                    self.providerList[x], self.inventoryList[self.providerList[x].item_id], self.daily_events,self.simpy_env.now,action[x]))
                    
             

                for inven in self.inventoryList:
                    daily_total_cost += inven.daily_inven_cost
                    print("id:", inven.item_id, "inven:", inven.current_level, "Cost:", inven.daily_inven_cost)
                    inven.daily_inven_cost = 0
                for production in self.productionList:
                    daily_total_cost += production.daily_production_cost
                    production.daily_production_cost = 0
                for procurement in self.procurementList:
                    daily_total_cost += procurement.daily_procurement_cost
                    procurement.daily_procurement_cost = 0
                daily_total_cost += self.sales.daily_selling_cost
                print("sales cost:", self.sales.daily_selling_cost)
                print("shortage:",self.sales.num_shortages)
                self.sales.daily_selling_cost=0
                reward =-daily_total_cost
                print("[Daily Total reward] ", reward)
                self.total_reward=reward
                
            self.current_observation = {
                'Day':np.array([self.day]),
                'DueDate':np.array([self.due_date]),
                'raw_material_inventory 1': np.array([self.inventoryList[1].current_level]),
                'raw_material_inventory 2': np.array([self.inventoryList[2].current_level]),
                'raw_material_inventory 3': np.array([self.inventoryList[3].current_level]),
                'WIP':np.array([self.inventoryList[4].current_level]),
                'product_inventory': np.array([self.inventoryList[0].current_level]),
                
                'current_demand': np.array([self.customer.order_history[-1]]),
            }  
            done=False
        
        else:
            
            if self.day%I[0]["CUST_ORDER_CYCLE"]==1:
                self.due_date=I[0]['DUE_DATE']
                if self.due_date==0:
                    pass
                else:
                    self.due_date= self.due_date-1
            if self.simpy_env.now%24==0:
                print(f"\nDay {self.day}:")
                for x in range(len(self.procurementList)):  
                    print("orders:",action[x])
                     # Procurements
                    self.simpy_env.process(self.procurementList[x].order(
                    self.providerList[x], self.inventoryList[self.providerList[x].item_id], self.daily_events,self.simpy_env.now,action[x]))

                for inven in self.inventoryList:
                    daily_total_cost += inven.daily_inven_cost
                    print("id:", inven.item_id, "inven:", inven.current_level, "Cost:", inven.daily_inven_cost)
                    inven.daily_inven_cost = 0
                for production in self.productionList:
                    daily_total_cost += production.daily_production_cost
                    production.daily_production_cost = 0
                for procurement in self.procurementList:
                    daily_total_cost += procurement.daily_procurement_cost
                    procurement.daily_procurement_cost = 0
                daily_total_cost += self.sales.daily_selling_cost
                print("sales cost:", self.sales.daily_selling_cost)
                print("shortage:",self.sales.num_shortages)
                self.sales.daily_selling_cost=0
                reward =-daily_total_cost
                print("[Daily Total reward] ", reward)
                self.total_reward=reward
                
            self.current_observation = {
                'Day':np.array([self.day]),
                'DueDate':np.array([self.due_date]),
                'raw_material_inventory 1': np.array([self.inventoryList[1].current_level]),
                'raw_material_inventory 2': np.array([self.inventoryList[2].current_level]),
                'raw_material_inventory 3': np.array([self.inventoryList[3].current_level]),
                'WIP':np.array([self.inventoryList[4].current_level]),
                'product_inventory': np.array([self.inventoryList[0].current_level]),
                
                'current_demand': np.array([self.customer.order_history[-1]]),
            }  
        done=False
        self.simpy_env.run(until=self.simpy_env.now+1) 
        return self.current_observation, reward, done, {}

        

    
def main():
    daily_events = []
   
    env = CustomSimPyEnv(daily_events,action_space)
    EPISODES = 10000 # 에피소드 수를 조정
    # model을 불러올려면 model=(A2C).load->
    model1 = A2C('MultiInputPolicy', env, verbose=0)
    '''
    model2 = PPO('MultiInputPolicy', env, verbose=0)
    model3 = PPO('MultiInputPolicy', env, verbose=0)
    '''
   
   # model=[model1,model2,model3]
    model1.learn(total_timesteps=24*SIM_TIME*EPISODES)
    history=np.array(env.total_reward_list)
            #재고제한 변경
    os.chdir('C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/A2C')
    history=pd.DataFrame(history)
    history.to_csv('A2C_csv.csv')
    model1.save("A2C_MODEL")
    
if __name__ == "__main__":
    main()
