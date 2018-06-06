#update num_train, num_test

from __future__ import division, print_function, absolute_import
import inp_file
import tensorflow as tf
import numpy as np
import time
start_time = time.time()

num_train =10 #no of training datasets in for each speed
num_test = 12 #no of test datasets

n_train = num_train*3 # no of datasets in training process
n_test = num_test*3 # no of data to be tested

#Getting inp data
accel = inp_file.read_inp(n_train,n_test,one_hot=False)

# Training Parameters
learning_rate = 0.001
num_steps = 4
batch_size = 40

# Network Parameters
num_input = 784 # accel data input (28*28)
num_classes = 9 # accel total classes , 3*3
dropout = 0.25 # Dropout, probability to drop a neuron

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['acdata']
        # accel data input is a 1-D vector of 784 features (28*28)
        # Reshape to match required format (2D)
        # Tensor input becomes 4-D: [Batch Size, Height, Width, Channel
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # Convolution Layer, 32 filters , kernel size 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling, kernel size 2, srtides 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # Convolution Layer, 64 filters , kernel size 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling, kernel size 2, srtides 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # Flatten conv2 to a 1-D vector for fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        # Fully connected layer
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (while training only)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        # Output layer - prediction of classes
        out = tf.layers.dense(fc1, n_classes)
        tf.Print(out,[out])

    return out

# Define the cnn model
def model_fn(features, labels, mode):
   
    # Building neural network
    # 2 distinct computation graphs for training and testing
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
    
    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    #pred_classes=tf.Print(pred_classes,[pred_classes])
    #pred_probas=tf.Print(pred_probas,[pred_probas])
    #pred_c=tf.Print(pred_c,[pred_c])
    #pred_p=tf.Print(pred_p,[pred_p])    

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # Define loss and optimizer, acccuracy
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
 
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})
    return estim_specs

#Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'acdata': accel.train.acdata}, y=accel.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
model.train(input_fn, steps=num_steps)

print('\n.......Training completed.......\n')
print('\nTime taken for training :',(time.time() - start_time)/60,'minutes(',time.time() - start_time,'sec.)\n')

# Get acdata from test set
test_acdata = accel.test.acdata[:n_test]

# Define the input function for testing
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'acdata': test_acdata}, shuffle=False)

# Test the model
preds = list(model.predict(input_fn))

nn = int(n_test/3)
#assumption: only accel values are used (x,y,z)
preds_edit = np.zeros(n_test)
#convert to real test labels
for i in range(n_test):
    for j in range(nn):
        if preds[i]==3*j+0 or preds[i]==3*j+1 or preds[i]==3*j+2:
            preds_edit[i] = j

preds_final = np.zeros(nn)
cnt=0
for i in range(0,n_test,3):
    avg = 0.0
    avg= (preds_edit[i] + preds_edit[i+1] + preds_edit[i+2] )/3
    avg = int(round(avg))
    preds_final[cnt] = avg
    cnt=cnt+1

#Display the predicted classes
for i in range(nn):
    print('Test data:',i+1, "   Predicted Speed:", int(preds_final[i] + 1))

#Finding accuracy
test_labels= np.array([1,2,3,1,3,1,2,2,1,3,1,2])
adn=0.0
for i in range(num_test):
    if preds_final[i]+1==test_labels[i] :
        adn=adn+1.0

print('\nAccuracy :',adn/num_test*100,'%') 
print('\nDone..\nTOTAL TIME TAKEN  :',(time.time() - start_time)/60,'minutes. (',time.time() - start_time,'seconds)\n')
