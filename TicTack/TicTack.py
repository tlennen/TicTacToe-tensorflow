import tensorflow as tf #when installing tensorflow, remember to use the 64-bit version of Python

# Using linear regression, we will estimate the cost of houses using 3 inputs and train the weights to better
# estimate the costs.

X = tf.placeholder(tf.float32, [None,3])# inputs as placeholders
Y = tf.placeholder(tf.float32, [None,1])

W = tf.get_variable("weights", [3,1], intializer = tf.random_normal_intializer())# parameters-> inputs
b = tf.get_variable("intercept", [1], intializer = tf.constant_intializer(0))

Y_hat=tf.matmul(X,W)+b # operations

cost = tf.reduce_mean(tf.square(Y_hat - Y)) # cost function

learning_rate = 0.05 # optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# TRAINING

NUM_EPOCHS = 3

with tf.Session() as sess:
    # intialize variables
    sess.run(tf.global_variables_intializer())

    # train
    for _ in range(NUM_EPOCHS):
        for(X_batch, Y_batch) in get_minibatches(X_train, Y_train, BATCH_SIZE):
            sess.run(optimizer, feed_dict = {X:X_batch,Y:Y_batch})

    # test
        Y_predicted =  sess.run(model, feed_dict = {X: X_test})

    squared_error = tf.reduce_mean(tf.square(Y_test, Y_predicted))