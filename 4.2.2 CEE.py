import numpy as np

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

# 동일한 계산
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

cee = cross_entropy_error(np.array(y), np.array(t))
print(cee)  # 0.510825457099

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

cee = cross_entropy_error(np.array(y), np.array(t))
print(cee)  # 2.30258409299