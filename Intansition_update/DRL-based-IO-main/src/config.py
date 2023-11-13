import random
#### Items #####################################################################
# ID: Index of the element in the dictionary
# TYPE: Product, Raw Material, WIP;
# NAME: Item's name or model;
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to providers [days]
# DEMAND_QUANTITY: Demand quantity for the final product [units] -> THIS IS UPDATED EVERY 24 HOURS (Default: 0)
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# REMOVE## LOT_SIZE_ORDER: Lot-size for the order of raw materials (Q) [units] -> THIS IS AN AGENT ACTION THAT IS UPDATED EVERY 24 HOURS
# HOLD_COST: Holding cost of the items [$/unit*day]
# PURCHASE_COST: Holding cost of the raw materials [$/unit]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# SETUP_COST_RAW: Setup cost for the ordering of the raw materials to a supplier [$/order]
# DELIVERY_COST: Delivery cost of the products [$/unit]
# DUE_DATE: Term of customer order to delivered [days]
# BACKORDER_COST: Backorder cost of products or WIP [$/unit]

#### Processes #####################################################################
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day]
# INPUT_TYPE_LIST: List of types of input materials or WIPs
# QNTY_FOR_INPUT_ITEM: Quantity of input materials or WIPs [units]
# OUTPUT: Output WIP or Product
# PROCESS_COST: Processing cost of the process [$/day]
# PROCESS_STOP_COST: Penalty cost for stopping the process [$/unit]

# Scenario 1
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
         "CUST_ORDER_CYCLE": 1,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 1,
         "SETUP_COST_PRO": 1,
         "DELIVERY_COST": 1,
         "DUE_DATE": 0,
         "BACKORDER_COST": 50},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1",
         "MANU_ORDER_CYCLE": 0,
         "SUP_LEAD_TIME":3,
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "SETUP_COST_RAW": 1}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [
    1], "OUTPUT": I[0], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2}}

'''
# Scenario 2
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",
         "CUST_ORDER_CYCLE": 1,
         "DEMAND_QUANTITY": 0,
         "HOLD_COST": 1,
         "SETUP_COST_PRO": 1,
         "DELIVERY_COST": 1,
         "DUE_DATE": 0,
         "BACKORDER_COST": 50},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1",
         "MANU_ORDER_CYCLE": 1,
         "SUP_LEAD_TIME": 0,
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "SETUP_COST_RAW": 1},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1",
         "MANU_ORDER_CYCLE": 1,
         "SUP_LEAD_TIME": 0,
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "SETUP_COST_RAW": 1},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2",
         "MANU_ORDER_CYCLE": 1,
         "SUP_LEAD_TIME": 0,
         "HOLD_COST": 1,
         "PURCHASE_COST": 2,
         "SETUP_COST_RAW": 1},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",
         "HOLD_COST": 1, }}
P = {0: {"ID": 0, "PRODUCTION_RATE": 2,
         "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [1],
         "OUTPUT": I[4],
         "PROCESS_COST": 2,
         "PROCESS_STOP_COST": 2},
     1: {"ID": 1, "PRODUCTION_RATE": 2,
         "INPUT_TYPE_LIST": [I[2], I[3], I[4]], "QNTY_FOR_INPUT_ITEM": [1, 1, 1],
         "OUTPUT": I[0],
         "PROCESS_COST": 2,
         "PROCESS_STOP_COST": 3}}
'''
# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
ACTION_SPACE = [0, 1, 2, 3, 4, 5]
# Scenario 1-1
# DQN
# BEST_PARAMS = {'learning_rate': 0.000039592987586748284,
#                'gamma': 0.946622398997048, 'batch_size': 256}
# PPO
# BEST_PARAMS = {'learning_rate': 0.004925696229801578,
#                'gamma': 0.9884195154449192, 'batch_size': 128}
# Scenario 1-2-1
# DQN
# BEST_PARAMS = {'learning_rate': 0.0001998587865387424,
#                'gamma':  0.99835157223117, 'batch_size': 64}
# Scenario 1-2-2
# PPO
BEST_PARAMS = {'learning_rate': 0.00012381151768747168,
               'gamma':  0.9292540359, 'batch_size': 256}

# Scenario 2-1-1
# PPO
# BEST_PARAMS = {'learning_rate': 0.000010551004230145447,'gamma': 0.9203615255548315, 'batch_size': 64}
# DDPG
# BEST_PARAMS = {'learning_rate':  0.000010064052848745629,
#                'gamma': 0.9265086581150084, 'batch_size': 32}
# Scenario 2-2-1
# PPO
# BEST_PARAMS = {'learning_rate': 0.0007062032033742153,
#                'gamma': 0.98118769052857, 'batch_size': 256}

# State space
# if this is not 0, the length of state space of demand quantity is not identical to INVEN_LEVEL_MAX
INVEN_LEVEL_MIN = 0
INVEN_LEVEL_MAX = 20  # Capacity limit of the inventory [units]
STATE_DEMAND = True  # True: Demand quantity is included in the state space

# Simulation
N_EPISODES = 1  # 3000
SIM_TIME = 5  # 200 [days] per episode
INIT_LEVEL = 10  # Initial inventory level [units]
# Stochastic demand
DEMAND_QTY_MIN = 0  # if this is not 0, the length of state space of demand quantity is not identical to DEMAND_QTY_MAX
DEMAND_QTY_MAX = 4
# DUE_DATE_MIN = 0  # if this is not 0, the length of state space of demand quantity is not identical to DUE_DATE_MAX
# DUE_DATE_MAX = 3


# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 50  # 50
N_EVAL_EPISODES = 100  # 100

# Print logs
PRINT_SIM_EVENTS = True
TENSORFLOW_LOGS = "C:/tensorboard_logs/"

DAILY_EVENTS = []
# Cost model
# If False, the total cost is calculated based on the inventory level for every 24 hours.
# Otherwise, the total cost is accumulated every hour.
HOURLY_COST_MODEL = True
