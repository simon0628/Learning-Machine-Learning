import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as op

from tools import *

data = load_txt('ex2data1.txt', ['exam1', 'exam2', 'admit'])

data = data.to_numpy()
X = data[:,:-1]
y = data[:,-1]

model = LogisticRegression(len(X[0]))
model.train_scipy(X,y)
print(model.test([45,85]))
