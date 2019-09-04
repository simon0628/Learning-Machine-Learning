import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from tools import *
# from tqdm import tqdm


def display_data(X,y, size = 10):
    choices = np.random.randint(len(y), size = size)
    X = X[choices]
    y = y[choices]
    plt.figure(figsize=(20,8))
    for index, (image, label) in enumerate(zip(X[0:size], y[0:size])):
        plt.subplot(size/5, 5, index + 1)
        plt.imshow(np.reshape(image, (20,20)), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % label, fontsize = 20)
    plt.show()

def deonehot(y):
    loc = np.argmax(y)
    return loc

def onehot(y):
    m = len(y)
    label2num = dict() # {label: num} e.g. {'cat':0, 'dog':1}
    num2label = dict() # {num: label}
    cnt = 0
    for yy in np.unique(y):
        label2num[yy] = cnt
        num2label[cnt] = yy
        cnt += 1
    y_onehot = np.zeros((m, cnt))

    for i in range(m):
        y_onehot[i][label2num[y[i][0]]] = 1
    return cnt, num2label, y_onehot


def display_prediction(X,y,y_pred, size = 10):
    choices = np.random.randint(len(y), size = size)
    X = X[choices]
    y = y[choices]
    y_pred = y_pred[choices]
    plt.figure(figsize=(20,8))
    for index, (image, label, pred) in enumerate(zip(X[0:size], y[0:size], y_pred[0:size])):
        plt.subplot(size/5, 5, index + 1)
        plt.imshow(np.reshape(image, (20,20)), cmap=plt.cm.gray)
        plt.title('digit=%d, prediction=%d' % (label, pred) , fontsize = 10)
    plt.show()


[X, y] = load_mat(expath(3) + "ex3data1.mat",['X','y'])

# display_data(X,y, 10)
K, y_dict, y_onehot = onehot(y)

print('data size: %d' % len(y))
x_train, x_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.25)

models = list()
for k in range(K):
    print('training on digit \'%d\'' % y_dict[k], end = ': ')
    model = LogisticRegression(len(X[0]))
    loss = model.train(x_train, y_train[:,k], 800, alpha = 0.3, lam = 10)
    print('loss =', loss)
    models.append(model)

match = list()
y_label = list()
y_preds = list()
for index, (image, label) in enumerate(zip(x_test, y_test)):
    y_pred_onehot = np.zeros(K)
    for k in range(K):
        y_pred_onehot[k] = models[k].test(image)
    
    digit = y_dict[deonehot(label)]
    prediction = y_dict[deonehot(y_pred_onehot)]
    y_label.append(digit)
    y_preds.append(prediction)
    match.append(1 if (digit == prediction) else 0)

# display_prediction(x_test, np.array(y_label), np.array(y_preds))

precision = sum(match) / len(match)
print('precision = ', precision)
