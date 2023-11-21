import random
import numpy as np
def make_data(start,num_cont):
    values = np.random.normal(5, 1, num_cont)
    # 값들을 주어진 범위로 클리핑
    values = np.clip(values, 2, 8)
    # 실수 값을 정수로 변환
    values = np.round(values).astype(int)
    return values
