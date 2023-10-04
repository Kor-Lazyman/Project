#### Items #####################################################################
# ID: Index of the element in the dictionary
# TYPE: Product, Raw Material, WIP;
# NAME: Item's name or model;
# INIT_LEVEL: Initial inventory level [units]
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to providers [days]
# DEMAND_QUANTITY: Demand quantity for the final product [units]
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# REMOVE ##  MANU_LEAD_TIME: The total processing time for the manufacturer to process and deliver the customer's order [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# LOT_SIZE_ORDER: Lot-size for the order of raw materials (Q) [units]
# HOLD_COST: Holding cost of the items [$/unit*day]
# SHORTAGE_COST: Shortage cost of items [$/unit]
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
'''
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",        "INIT_LEVEL":0, "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 10,                                           "HOLD_COST": 10, "SHORTAGE_COST": 10,                     "SETUP_COST_PRO": 50, "DELIVERY_COST": 10, "DUE_DATE": 5, "BACKORDER_COST": 5},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1", "INIT_LEVEL": 10, "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME": 1, "ORDER": [0, 1] ,"LOT_SIZE":10,"HOLD_COST": 10, "SHORTAGE_COST": 8, "PURCHASE_COST": 3,  "SETUP_COST_RAW": 20}}
P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [
    1], "OUTPUT": I[0], "PROCESS_COST": 10, "PROCESS_STOP_COST": 2}}

'''
# Scenario 2
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "INIT_LEVEL": 0, "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 5,                                           "HOLD_COST": 8,                    "SETUP_COST_PRO": 20, "DELIVERY_COST": 10, "DUE_DATE": 7, "BACKORDER_COST":100},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1", "INIT_LEVEL": 10, "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME": 1, "ORDER": [0, 1],"LOT_SIZE": 20, "HOLD_COST": 3, "SHORTAGE_COST": 0, "PURCHASE_COST": 5,  "SETUP_COST_RAW": 5},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1", "INIT_LEVEL": 10, "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME":1, "ORDER": [0, 1],"LOT_SIZE": 20, "HOLD_COST": 4, "SHORTAGE_COST": 0, "PURCHASE_COST": 5,  "SETUP_COST_RAW": 5},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2", "INIT_LEVEL": 10, "MANU_ORDER_CYCLE": 1,                        "SUP_LEAD_TIME": 1, "ORDER": [0, 1],"LOT_SIZE": 20, "HOLD_COST": 5, "SHORTAGE_COST": 0, "PURCHASE_COST": 5,  "SETUP_COST_RAW": 5},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",            "INIT_LEVEL": 0,                                                                                         "HOLD_COST": 7, "SHORTAGE_COST": 0}}

P = {0: {"ID": 0, "PRODUCTION_RATE": 3, "INPUT_TYPE_LIST": [I[1]]            , "QNTY_FOR_INPUT_ITEM": [1]    , "OUTPUT": I[4], "PROCESS_COST": 20, "PROCESS_STOP_COST": 2},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[2], I[3], I[4]], "QNTY_FOR_INPUT_ITEM": [1,2,1], "OUTPUT": I[0], "PROCESS_COST":20, "PROCESS_STOP_COST": 3}}
'''
# Print logs
PRINT_SIM_EVENTS = True
PRINT_DQN = True

COST_VALID = False
VISUAL = False
SPECIFIC_HOLDING_COST = False
EventHoldingCost = []
'''
# Simulation
SIM_TIME = 100# [days]
# INITIAL_INVENTORY = 100  # [units]EPISODES = 1
total_cost_per_day = []
batch_size = 32
action_space = []
values = [0, 10, 20]
