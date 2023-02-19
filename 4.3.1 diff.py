import numpy as np
import matplotlib.pylab as plt

# 4.3.1 미분
# 나쁜 구현 예
def numerical_diff_bad(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h
# h값이 너무 작아 반올림 오차를 일으킬 수 있음 10e-4정도가 적당하다고 알려짐
# 전방 차분에서는 차분이 0이 될 수 없어 오차가 발생
#  -> 오차를 줄이기 위해 중심 차분을 사용

def numerical_diff(f, x):
    h = 10e-4
    return (f(x + h) - f(x - h)) / (2 * h)

