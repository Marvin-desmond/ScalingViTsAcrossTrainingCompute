import numpy as np 
from numpy import max, sum, exp 

def softmax(scores):
    max_vals = np.expand_dims(max(scores, axis=-1), -1)
    den = np.expand_dims(sum(np.exp(scores-max_vals), axis=-1),-1)
    return exp(scores-max_vals)/den



