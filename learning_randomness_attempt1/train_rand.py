import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


datacount = 1000
seed_start = 0 

trainx = []
trainy = []

for i in range(datacount):
    sdd = seed_start + i
    
    # set the seed for random generator
    np.random.seed(sdd)        
    
    #generate a random number between 0-1
    rad = np.random.rand()   
    
    trainx.append(sdd)
    trainy.append(rad)
	
# Divide data by half for testing and training
train_X = np.asarray(trainx[0:int(datacount/2)]) 
train_Y = np.asarray(trainy[0:int(datacount/2)])
test_X  = np.asarray(trainx[int(datacount/2):])
test_Y  = np.asarray(trainy[int(datacount/2):])

#Lets do some Standardization on X to scale it down
#otherwise x range is 0:1000 vs 0:1 in y

xmax = max(train_X)
stdx = np.std(train_X)

tmax = max(test_X)
stdt = np.std(test_X)

train_X = (train_X - xmax) / stdx

#test_X  = (test_X - xmax) / stdx
test_X  = (test_X - tmax) / stdt



plt.scatter(train_X,train_Y)
plt.show()

plt.scatter(test_X,test_Y)
plt.show()


rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 10

n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn()/100, name="weight")
b = tf.Variable(rng.randn()/100, name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Fit all training data
for epoch in range(training_epochs):
    for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: x, Y: y})

    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", sess.run(W), "b=", sess.run(b))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


y_res = sess.run(pred, feed_dict={X: train_X})

plt.scatter(train_X,train_Y)
plt.plot(train_X,y_res,linewidth=4, color='r')

plt.show()


rng = np.random

learning_rate = 0.01
training_epochs = 200
display_step = 50

n_samples = train_X.shape[0]


X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(rng.randn(1,16), name="weight1",dtype="float")
b1 = tf.Variable(rng.randn(16), name="bias1",dtype="float")

fc1 = tf.nn.relu(tf.matmul(X, W1) + b1)       #out1 = tf.add(tf.multiply(X, W1), b1)


W2 = tf.Variable(rng.randn(16,32), name="weight2",dtype="float")
b2 = tf.Variable(rng.randn(32), name="bias2",dtype="float")

fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

W3 = tf.Variable(rng.randn(32,1), name="weight3",dtype="float")
b3 = tf.Variable(rng.randn(1), name="bias3",dtype="float")

pred = tf.matmul(fc2, W3) + b3                  #pred = tf.nn.relu(tf.matmul(X, W1) + b1)


cost      = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()
loss_history = []

sess = tf.Session()
sess.run(init)

# Fit all training data
for epoch in range(training_epochs):
    for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: x.reshape([1,1]), Y: y.reshape([1,1])})

    c = sess.run(cost, feed_dict={X: train_X.reshape([int(datacount/2),1]), Y:train_Y.reshape([int(datacount/2),1])})     
    loss_history.append(c)
  

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: train_X.reshape([int(datacount/2),1]), Y: train_Y.reshape([int(datacount/2),1])})
print("Training cost=", training_cost, '\n')


plt.plot(loss_history)
plt.show()


y_res = sess.run(pred, feed_dict={X: train_X.reshape([int(datacount/2),1])})

plt.scatter(train_X,train_Y)
plt.plot(train_X,y_res,linewidth=4, color='r')

plt.show()

y_res_test = sess.run(pred, feed_dict={X: test_X.reshape([int(datacount/2),1])})

plt.scatter(test_X,test_Y)
plt.plot(test_X,y_res_test,linewidth=4, color='r')

plt.show()
