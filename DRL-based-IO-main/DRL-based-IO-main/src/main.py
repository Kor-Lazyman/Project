from pyexpat import model
from re import T
import environment as Env
import numpy as np
import random
from visualization import *
import time 
from config import *
import gym
from stable_baselines3 import DQN
import gym
from gym import spaces
from tensorboardX import SummaryWriter

# TensorBoard 로깅을 위한 디렉토리 경로 설정
log_dir="logs/"

class CustomSimPyEnv(gym.Env):
    def __init__(self, daily_events):
        self.action_spaces=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        self.actions=[]
        self.due_date=0
        self.day=0
        self.daily_events = daily_events
        super(CustomSimPyEnv, self).__init__()
        self.time = 0
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict({
            'Day':spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'DueDate':spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'raw_material_inventory 1_1': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'raw_material_inventory 2_1': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'raw_material_inventory 2_2': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'WIP':spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'product_inventory': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'current_demand': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'incoming_raw_materials 1_1': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'incoming_raw_materials 2_1': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'incoming_raw_materials 2_2': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)
        })
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = Env.create_env(I, P, daily_events)
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.current_observation = {
            'Day':np.array([self.day]),
            'DueDate':np.array([self.due_date]),
            'raw_material_inventory 1_1': np.array([I[1]["INIT_LEVEL"]]),
            'raw_material_inventory 2_1': np.array([I[2]["INIT_LEVEL"]]),
            'raw_material_inventory 2_2': np.array([I[3]["INIT_LEVEL"]]),
            'WIP':np.array([I[4]["INIT_LEVEL"]]),
            'product_inventory': np.array([self.inventoryList[0].current_level]),
            'current_demand': np.array([0]),
            'incoming_raw_materials 1_1': np.array([0]),
            'incoming_raw_materials 2_1': np.array([0]),
            'incoming_raw_materials 2_2':np.array([0]),
        }
        self.episode_length = 365

    def reset(self):
        self.due_date=0
        daily_events = []
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = Env.create_env(I, P, daily_events)
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.current_observation = {
            'Day':np.array([self.day]),
            'DueDate':np.array([self.due_date]),
            'raw_material_inventory 1_1': np.array([I[1]["INIT_LEVEL"]]),
            'raw_material_inventory 2_1': np.array([I[2]["INIT_LEVEL"]]),
            'raw_material_inventory 2_2': np.array([I[3]["INIT_LEVEL"]]),
            'WIP':np.array([I[4]["INIT_LEVEL"]]),

            'product_inventory': np.array([0]),
            'current_demand': np.array([0]),
            'incoming_raw_materials 1_1': np.array([0]),
            'incoming_raw_materials 2_1': np.array([0]),
            'incoming_raw_materials 2_2':np.array([0]),
        }
        return self.current_observation

    def step(self, action):
            
            reward=0
            if self.day%I[0]["CUST_ORDER_CYCLE"]==1:
                self.due_date=I[0]['DUE_DATE']
            daily_total_cost = 0
            if (self.time) % 24 == 0:
                for x in range(len(self.inventoryList)):
                    if self.inventoryList[x].current_level>50:
                        reward=reward-100*(self.inventoryList[x].current_level-50)
                        self.inventoryList[x].current_level=50
                self.day=self.day+1
                if self.due_date==0:
                    pass
                else:
                    self.due_date= self.due_date-1
                

                for x in range(len(self.procurementList)):  
                    print(action)
                    self.procurementList[x].action=self.action_spaces[action][x]
                    
                    
                print(f"\nDay {(self.time) // 24+1}:")
                if self.inventoryList[0].current_level>50:
                    reward=reward-50*(self.inventoryList[0].current_level-50)
                    self.inventoryList[0].current_level=50
                for inven in self.inventoryList:
                    daily_total_cost += inven.daily_inven_cost
                    print("id:", inven.item_id, "inven:", inven.current_level, "Cost:", inven.daily_inven_cost)
                    inven.daily_inven_cost = 0
                for production in self.productionList:
                    daily_total_cost += production.daily_production_cost
                    print("production cost", production.daily_production_cost)
                    production.daily_production_cost = 0
                for procurement in self.procurementList:
                    daily_total_cost += procurement.daily_procurement_cost
                    print("procurement cost:", procurement.daily_procurement_cost)
                    procurement.daily_procurement_cost = 0
                daily_total_cost += self.sales.daily_selling_cost
                self.sales.daily_selling_cost = 0
                print("sales cost:", self.sales.daily_selling_cost)
                print("[Daily Total Cost] ", daily_total_cost)
                reward =reward -daily_total_cost
            self.current_observation = {
                'Day':np.array([self.day]),
                'DueDate':np.array([self.due_date]),
                'raw_material_inventory 1_1': np.array([self.inventoryList[1].current_level]),
                'raw_material_inventory 2_1': np.array([self.inventoryList[3].current_level]),
                'raw_material_inventory 2_2': np.array([self.inventoryList[3].current_level]),
                'WIP':np.array([self.inventoryList[4].current_level]),
                'product_inventory': np.array([self.inventoryList[0].current_level]),
                
                'current_demand': np.array([self.customer.order_history[-1]]),
                'incoming_raw_materials 1_1': np.array([I[1]["LOT_SIZE"] * self.action_spaces[action][0]]),
                'incoming_raw_materials 2_1': np.array([I[2]["LOT_SIZE"] * self.action_spaces[action][1]]),
                'incoming_raw_materials 2_2':np.array([I[3]["LOT_SIZE"] * self.action_spaces[action][2]]),
            }  
            done = False
        
                    
           
            return self.current_observation, reward, done, {}
        

def epsilon_greedy_action(model, obs, episode):
    # 에피소드가 진행됨에 따라 epsilon 값을 조정하여 탐색 감소
    epsilon = ((10000 - episode) / 100) 
    if np.random.rand() < epsilon:
        # 랜덤 액션 선택
        return random.randint(0,7)
    else:
        return model.predict(obs, deterministic=True)[0]
        # 모델의 액션 선택           
       
    
def main():
    daily_events = []
    epsilon = 1
    env = CustomSimPyEnv(daily_events)
    EPISODES = 10000  # 에피소드 수를 조정
    model = DQN('MultiInputPolicy', env, verbose=1)
    writer = SummaryWriter(log_dir)

    for episode in range(EPISODES):
        episode_start_time = time.time()
        obs = env.reset()
        total_reward = 0
        action=0
        for i in range(SIM_TIME * 24):
            
            env.simpy_env.run(until=i + 1)
            
            env.time=i
           
            if (i % 24) == 0:
                action =  epsilon_greedy_action(model, obs, episode)
            obs, reward, done, dummy = env.step(action)
            total_reward = reward + total_reward
            
        episode_end_time = time.time()
        episode_training_time = episode_end_time - episode_start_time
        writer.add_scalar("Total reward", total_reward, episode)
        writer.add_scalar("Training Time per Episode (seconds)", episode_training_time)
        model.learn(total_timesteps=SIM_TIME)
        print(daily_events)
        print(f'Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}')
        total_reward=0
    writer.close()
    model.save("DQN_MODEL")
    print(daily_events)
if __name__ == "__main__":
    main()
'''
        
        if self.time== SIM_TIME * 24:
            self.day=self.day+1
            if self.due_date==0:
                pass
            else:
                self.due_date= self.due_date-1
        
            self.current_observation = {
            'Day':np.array([self.day]),
            'DueDate':np.array([self.due_date]),
            'raw_material_inventory': np.array([self.inventoryList[0].current_level]),
            'product_inventory': np.array([self.inventoryList[1].current_level]),
            'current_demand': np.array([self.customer.order_history[-1]]),
            'incoming_raw_materials': np.array([I[1]["LOT_SIZE"] * action])
            }
            reward = 0
            done = True
        
        else:
            
'''
