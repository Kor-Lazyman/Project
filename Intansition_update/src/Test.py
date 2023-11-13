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
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import gym
from gym import spaces
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import os
import test_data
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
        self.action_space=spaces.Box(low=0, high=4, shape=(3,), dtype=np.float32)
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
            'product_inventory': np.array([I[0]["INIT_LEVEL"]]),
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
                if self.due_date==0:
                    pass
                else:
                    self.due_date= self.due_date-1
            
        if self.simpy_env.now%24==0:
                print(f"\nDay {self.day}:")
                for x in range(len(self.procurementList)):  
                    print("orders:",round(action[x]))
                     # Procurements
                    self.simpy_env.process( self.procurementList[x].order(
                    self.providerList[x], self.inventoryList[self.providerList[x].item_id], self.daily_events,self.simpy_env.now,round(action[x])))
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
    action_space=[]
    test_datas=test_data.make_data()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                action_space.append([x,y,z])
    env = CustomSimPyEnv(daily_events,action_space)
    
    # 에피소드 수를 조정
    # model을 불러올려면 model=(A2C).load->
    Model_SAC = SAC.load('C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/SAC/SAC_MODEL.zip')
    Model_A2C= A2C.load('C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/A2C/A2C_MODEL.zip')
    Model_PPO= PPO.load('C:/Users/User/Downloads/DRL-based-IO-main/DRL-based-IO-main/src/logs/PPO/PPO_MODEL.zip')
    '''
    model2 = PPO('MultiInputPolicy', env, verbose=0)
    model3 = PPO('MultiInputPolicy', env, verbose=0)
    '''
    models=[Model_SAC,Model_PPO,Model_A2C]
    totalreward=0
    total_reward_list=[[],[],[],[]]
    shortage=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    db=[]
    for z in range(len(models)+1):
        inven_temp=[0,0,0,0,0]
        for x in range(200): 
            env.customer.test_data=test_datas
            totalreward=0
            env.reset()
            env.customer.test_case=x
            if z==0 or z==1:
                env.action_space=spaces.Box(low=0, high=4, shape=(3,), dtype=np.float32)
            elif z==2:
                env.action_space=spaces.MultiDiscrete([5,5,5])
        
            
        
            temp=[]
            
            for y in range(0,100):
                
                if z==3:
                    action=[random.randint(0,5),random.randint(0,5),random.randint(0,5)]
                    env.current_observation,reward,done,dummy=env.step(action)
                else:
                    action=models[z].predict(env.current_observation)
                    env.current_observation,reward,done,dummy=env.step(action[0])
               
                for inven in env.inventoryList:
                    inven_temp[inven.item_id]=inven.current_level/(100*200)+inven_temp[inven.item_id]
                    
       
                totalreward=totalreward+reward
                if y%7==0 and y!=0:
                    temp.append(env.sales.num_shortages)
            if z<3:
                
                for i in range(len(shortage[0])):    
                    shortage[z][i]=(shortage[z][i]+temp[i]/200)
                   
            total_reward_list[z].append(-totalreward/100)
        db.append(inven_temp)

    plt.figure(figsize=(20, 10))


    plt.plot(total_reward_list[0],label="SAC")
    plt.plot(total_reward_list[1],label="PPO")
    plt.plot(total_reward_list[2],label="A2C")
    plt.plot(total_reward_list[3],label="Random")
    plt.xlim(0,200)
    plt.xticks([0,50,100,150,200],fontsize=45)
    plt.yticks(fontsize=45)
    plt.xlabel("test case",fontsize=45)
    plt.ylim(200,600)
    plt.yticks([200,400,600],fontsize=45)
    plt.ylabel("Cost",fontsize=45)
    plt.legend(loc = 'upper right',fontsize=30)
    plt.show()
    plt.plot(shortage[0],label="SAC")
    plt.plot(shortage[1],label="PPO")
    plt.plot(shortage[2],label="A2C")
    plt.xlim(0,14)
    plt.xlabel("Check for x th shortage",fontsize=20)
    plt.xticks([0,7,14],fontsize=20)
    plt.ylabel("shortage",fontsize=20)
    plt.yticks([0,5,10,15,20],fontsize=20)
    plt.show()
    plt.legend(loc = 'upper left',fontsize=20)
    bar_width = 0.2
    plt.bar(range(5), db[0], 0.2, label='SAC')
    plt.bar([i + bar_width for i in range(5)], db[1], 0.2, label='PPO')
    plt.bar([i + bar_width * 2 for i in range(5)], db[2], 0.2, label='A2C')
    plt.ylim(0,20)
    plt.xticks(fontsize=15)
    plt.yticks([0,5,10,15,20],fontsize=15)
    plt.xlabel("Item_ID",fontsize=15)
    plt.ylabel("Average Inventory",fontsize=15)
    plt.legend(loc = 'upper left',fontsize=10)

    plt.legend(fontsize=15)
if __name__ == "__main__":
    main()