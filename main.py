import data_service as ds
import plot_service as plot
import tensorflow as tf
import cnn_service as cnn
from time import time
import numpy as np

count_of_single_letter = 50

train_images, train_labels = ds.load_train_data_fixed(limit_for_single_letter=count_of_single_letter)

# Parameters
learning_rate = 0.001
batch_size = 128
learn_iterations =  500
epoch_step = 2
# Network Parameters
classes_count = 29
input_image_size = 64*64*3
dropout = 0.15

model_parameters = {
'filter_c1': tf.Variable(tf.random_normal([3,3,3,32])),
'filter_c2' : tf.Variable(tf.random_normal([3,3,32,64])),
'filter_c3' : tf.Variable(tf.random_normal([3,3,64,64])),
'weights_f' : tf.Variable(tf.random_normal([8*8*64, 512])),
'weights_l' :  tf.Variable(tf.random_normal([512, classes_count ])),
'bias_l' : tf.Variable(tf.random_normal([classes_count])),
'bias_f' :  tf.Variable(tf.random_normal([512])),
'bias_c1' : tf.Variable(tf.random_normal([32])),
'bias_c2' : tf.Variable(tf.random_normal([64])),
'bias_c3' : tf.Variable(tf.random_normal([64]))}


# tf Graph input
x = tf.placeholder(tf.float32, [None, input_image_size])
y = tf.placeholder(tf.float32, [None, classes_count])
keep_probability = tf.placeholder(tf.float32) #dropout (keep probability)

# create model
logits = cnn.cnn_model(input_layer = x, dropout = keep_probability, param = model_parameters)

# create loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
prediction = tf.nn.softmax(logits)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# evaluate model
#correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy  = tf.metrics.accuracy(predictions = prediction, labels = y) 

######
# Init TF variables
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

losses = list()
accuracies = list()

# startup session
with tf.Session() as sess:
    sess.run(init)
    sess.run(init_local)

    
    step = 0
    epoch = 0
    start_global = time()
    start_epoch = time()

    print("######################################################")
    print("Optimization started: {}".format(plot.time_format(start_global, 'dt')))
    print("######################################################")
    
    # For every learn iteration
    while step <= learn_iterations:

        # get random batch data
        random_images, random_labels = ds.get_random_batch(train_images, train_labels, batch_size)
        
        # optimize
        start_optimalization = time()
        sess.run(optimizer, feed_dict={x: random_images, y: random_labels, keep_probability: dropout})
        stop_optimalization = time()
        print("#{} optimization step: {}".format(step, plot.time_interval(start_optimalization ,stop_optimalization)))            
        
        # For every epoch
        if step % epoch_step == 0:
            
            # get learn accuracy
            acc = sess.run(accuracy, feed_dict={x: random_images, y: random_labels, keep_probability: dropout})
            accuracies.append(acc)
            
            # get batch loss
            batch_loss = sess.run(loss, feed_dict={x: random_images, y: random_labels, keep_probability: dropout})
            losses.append(batch_loss)
            
            # print epoch results
            stop_epoch = time()
            print("@@@ Epoch {} finished during {} - {}.".format(epoch, plot.time_format(start_epoch, 't'), plot.time_format(stop_epoch, 't')))
            print("    Network results:")
            print("    Itertion " + str(step) + ": batch loss = " + "{0:.5f}".format(batch_loss) +" accuracy: "+ str(acc[1])) 
            
            epoch += 1
            start_epoch = stop_epoch
        
        step += 1     
          
    # print global results
    stop_global = time()
    print("######################################################")
    print("Optimization finished. Started: {}, finished: {}, duration {}".format(plot.time_format(start_global, 'dt'), plot.time_format(stop_global, 'dt'), plot.time_interval(start_global, stop_global)))
    print("######################################################")
    
    # do test run
    test_size = min(400, train_images.shape[0])
    test_X = train_images[0:test_size,:]
    test_Y = train_labels[0:test_size,:]
    # Calculate accuracy 
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_X, y: test_Y, keep_probability: dropout}))