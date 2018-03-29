import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


rng = np.random

learning_rate = 0.01
training_epochs = 2000
display_step = 10

trainx = []
trainy = []

datacount   = 10000
train_count = 5000

seed_start = 0 

for i in range(datacount):
    #sdd = int(time.time())
    sdd = seed_start + i
    np.random.seed(sdd)
    
    #rad = np.random.randint(-2**31,2**31)
    
    rad = np.random.rand()    #np.random.randint(0,10)
    
    trainx.append(sdd)
    trainy.append(rad)





train_X = np.asarray(trainx[0:train_count])
train_Y = np.asarray(trainy[0:train_count])

test_X = np.asarray(trainx[train_count:])
test_Y = np.asarray(trainy[train_count:])

xmax = max(train_X)
stdx = np.std(train_X)

#ymax = max(train_Y)
#stdy = np.std(train_Y)

train_X = (train_X - xmax) / stdx
#train_Y = (train_Y - ymax) / stdy



plt.scatter(train_X,train_Y)
plt.show()






















