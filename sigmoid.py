import math
def sigmoid(x):
    return 1 /(1+math.exp(-x))

print(sigmoid(0))
print(sigmoid(1))
print(sigmoid(-1))

#or use the numpy library
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([0,1,-1])
print(sigmoid(x))