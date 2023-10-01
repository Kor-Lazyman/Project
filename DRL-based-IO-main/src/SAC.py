from pyexpat import model
from re import T
import environment as Env
import numpy as np
import random
from visualization import *
import time 
from config import *
import gym
from stable_baselines3 import DDPG
import gym
from gym import spaces
from tensorboardX import SummaryWriter

# TensorBoard 로깅을 위한 디렉토리 경로 설정
log_dir="logs/"

class CustomSimPyEnv(gym.Env):
    def __init__(self, daily_events):
        self.daily_events = daily_events
        super(CustomSimPyEnv, self).__init__()
        self.time = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            'raw_material_inventory': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'product_inventory': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'current_demand': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            'incoming_raw_materials': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)
        })
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = Env.create_env(I, P, daily_events)
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.current_observation = {
            'raw_material_inventory': np.array([self.inventoryList[1].current_level]),
            'product_inventory': np.array([self.inventoryList[0].current_level]),
            'current_demand': np.array([0]),
            'incoming_raw_materials': np.array([0])
        }
        self.episode_length = 365

    def reset(self):
        daily_events = []
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = Env.create_env(I, P, daily_events)
        Env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        self.current_observation = {
            'raw_material_inventory': np.array([0]),
            'product_inventory': np.array([0]),
            'current_demand': np.array([0]),
            'incoming_raw_materials': np.array([0])
        }
        self.time = 0
        return self.current_observation

    def step(self, action):
        reward=0
        if self.time== SIM_TIME * 24:
            self.current_observation = {
            'raw_material_inventory': np.array([self.inventoryList[0].current_level]),
            'product_inventory': np.array([self.inventoryList[1].current_level]),
            'current_demand': np.array([self.customer.order_history[-1]]),
            'incoming_raw_materials': np.array([I[1]["LOT_SIZE"] * action])
            }
            reward = 0
            done = True
        
        else:
            
            self.simpy_env.run(until=self.time + 1)
            daily_total_cost = 0
            if (self.time + 1) % 24 == 0:
                
                if self.inventoryList[1].current_level+self.current_observation['incoming_raw_materials'][0]>50:
                    action=0 
                self.procurementList[0].action = action
                print(f"\nDay {(self.time + 1) // 24}:")
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
                'raw_material_inventory': np.array([self.inventoryList[0].current_level]),
                'product_inventory': np.array([self.inventoryList[1].current_level]),
                'current_demand': np.array([self.customer.order_history[-1]]),
                'incoming_raw_materials': np.array([I[1]["LOT_SIZE"] * action])
            }
            done = False
        self.time += 1
        
                    
           
        return self.current_observation, reward, done, {}
def epsilon_greedy_action(model, obs, episode, epsilon):
    # 에피소드가 진행됨에 따라 epsilon 값을 조정하여 탐색 감소
    epsilon = ((1000 - episode) / 10) * (epsilon + 1)
    if np.random.rand() < epsilon:
        # 랜덤 액션 선택
        return random.randint(0, 2)
    else:
        # 모델의 액션 선택
        return model.predict(obs, deterministic=True)[0]
def main():
    daily_events = []
    epsilon = 1
    env = CustomSimPyEnv(daily_events)
    EPISODES = 1000  # 에피소드 수를 조정
    model = DDPG('MultiInputPolicy', env, verbose=1)
    writer = SummaryWriter(log_dir)

    for episode in range(EPISODES):
        episode_start_time = time.time()
        obs = env.reset()
        total_reward = 0
        for i in range(SIM_TIME * 24):
            env.time = i
            if (i % 24) == 0:
                action = epsilon_greedy_action(model, obs, episode, epsilon)
            obs, reward, done, dummy = env.step(action)
            total_reward = reward + total_reward

        episode_end_time = time.time()
        episode_training_time = episode_end_time - episode_start_time
        writer.add_scalar("Total reward", total_reward, episode)
        writer.add_scalar("Training Time per Episode (seconds)", episode_training_time, episode)
        model.learn(total_timesteps=SIM_TIME)
        print(daily_events)
        print(f'Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}')

    writer.close()
    model.save("A2C_MODEL")

if __name__ == "__main__":
    main()