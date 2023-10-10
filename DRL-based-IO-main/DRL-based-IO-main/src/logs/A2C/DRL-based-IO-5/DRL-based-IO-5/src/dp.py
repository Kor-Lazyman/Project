from config import *
from DQN import *
import environment as env
import numpy as np
import random

simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList = env.create_env(
    I, P)
