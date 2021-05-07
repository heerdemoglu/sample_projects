import numpy as np
from sigmoid import *

def calculate_hypothesis(X, theta, i):
    return sigmoid(np.dot(X[i], theta))
    
