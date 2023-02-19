import numpy as np

# E = 1/2 * ∑ _k (yk-tk)²
# yk : 신경망의 출력
# tk : 정답 레이블
# k : 데이터의 차원 수
def mean_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 정답 : 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# ex1 '2'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

mse = sum_squares_error(np.array(y), np.array(t))
print(mse)  # 0.0975

# ex2 '7'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mse = sum_squares_error(np.array(y), np.array(t))
print(mse)  # 0.5975