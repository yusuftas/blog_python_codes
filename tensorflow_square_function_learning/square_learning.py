import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


def trainingFunction(sess,optimizer,cost,train_X,train_Y,display_step,training_epochs):
    
    loss_history = [] 
    data_count_train = train_X.shape[0] 
    
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x.reshape([1,1]), Y: y.reshape([1,1])})

        c = sess.run(cost, feed_dict={X: train_X.reshape([int(data_count_train),1]), Y:train_Y.reshape([int(data_count_train),1])})     
        loss_history.append(c)


        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X.reshape([int(datacount/2),1]), Y: train_Y.reshape([int(datacount/2),1])})
    print("Training cost=", training_cost, '\n')
    
    return loss_history
	
	

datacount = 1000 

#linearly spaced 1000 values between 0 and 1
trainx = np.linspace(0, 1, 1000)                      #trainx = np.random.rand(datacount)
trainy = np.asarray([v**2 for v in trainx])


#shuffle the X for a good train-test distribution
p = np.random.permutation(datacount)
trainx = trainx[p]
trainy = trainy[p]

    
# Divide data by half for testing and training
train_X = np.asarray(trainx[0:int(datacount/2)]) 
train_Y = np.asarray(trainy[0:int(datacount/2)])
test_X  = np.asarray(trainx[int(datacount/2):])
test_Y  = np.asarray(trainy[int(datacount/2):])

#plot the train and test data
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

#ax1.scatter(train_X,train_Y)
#ax1.set_title('Training data')
#ax2.scatter(test_X,test_Y, color='r')
#ax2.set_title('Testing data')

#lets first create the tensorflow environment we will work in
rng = np.random
learning_rate = 0.1
n_samples = train_X.shape[0]


X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(rng.randn(1,128)*2, name="weight1",dtype="float")
b1 = tf.Variable(rng.randn(128)*2, name="bias1",dtype="float")

fc1 = tf.nn.relu(tf.matmul(X, W1) + b1)       #out1 = tf.add(tf.multiply(X, W1), b1)


#W2 = tf.Variable(rng.randn(16,32), name="weight2",dtype="float")
#b2 = tf.Variable(rng.randn(32), name="bias2",dtype="float")

#fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

W3 = tf.Variable(rng.randn(128,1)*2, name="weight3",dtype="float")
b3 = tf.Variable(rng.randn(1)*2, name="bias3",dtype="float")

pred = tf.matmul(fc1, W3) + b3                  #pred = tf.nn.relu(tf.matmul(X, W1) + b1)


cost      = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


training_epochs = 1000
display_step = 50

#refresh the session and initialize the weights
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#use the training function we created to train the model
loss_history = trainingFunction(sess,optimizer,cost,train_X,train_Y,display_step,training_epochs)


f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2)

y_res = sess.run(pred, feed_dict={X: train_X.reshape([int(datacount/2),1])})
y_res_test = sess.run(pred, feed_dict={X: test_X.reshape([int(datacount/2),1])})

ax1.scatter(train_X,train_Y)
ax1.set_title('Training actual')
ax2.scatter(train_X,y_res, color='r')
ax2.set_title('Training predicted')

ax3.scatter(test_X,test_Y)
ax3.set_title('Testing actual')
ax4.scatter(test_X,y_res_test, color='r')
ax4.set_title('Testing predicted')

plt.show()
