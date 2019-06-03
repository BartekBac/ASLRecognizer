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
batch_size = 300
training_iters =  50
display_step = 10
# Network Parameters
n_classes = 29
n_input = 64*64*3
dropout = 0.75 # Dropout, probability to keep units

model_parameters = {
'filter_c1': tf.Variable(tf.random_normal([5,5,3,32])),
'filter_c2' : tf.Variable(tf.random_normal([5,5,32,64])),
'weights_f' : tf.Variable(tf.random_normal([16*16*64, 1024])),
'weights_l' :  tf.Variable(tf.random_normal([1024, n_classes ])),
'bias_l' : tf.Variable(tf.random_normal([n_classes])),
'bias_f' :  tf.Variable(tf.random_normal([1024])),
'bias_c1' : tf.Variable(tf.random_normal([32])),
'bias_c2' : tf.Variable(tf.random_normal([64]))}


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Construct model
prediction = cnn.cnn_model(input_layer = x, dropout = keep_prob, param = model_parameters)
# prepare loss and optimizer variables
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
prediction = tf.nn.softmax(prediction)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# evaluate model
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

######
# Initializing the variables
init = tf.global_variables_initializer()

losses = list()
accuracies = list()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    step = 0
    epoch = 0
    start_global = time()
    start_epoch = time()

    print("######################################################")
    print("Optimization started: {}".format(plot.time_format(start_global, 'dt')))
    print("######################################################")
    
 
    # Keep training until reach max iterations
    while step <= training_iters:

        random_images, random_labels = ds.get_random_batch(train_images, train_labels, batch_size)
        
        # Fit training using batch data
        start_optimalization = time()
        sess.run(optimizer, feed_dict={x: random_images, y: random_labels, keep_prob: dropout})
        stop_optimalization = time()
        print("#{} optimization step: {}".format(step, plot.time_interval(start_optimalization ,stop_optimalization)))            
        
        if step % display_step == 0:
            
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: random_images, y: random_labels, keep_prob: 1.}) #keep_prob zmienione z 1. na dropout
            accuracies.append(acc)
            
            # Calculate batch loss
            batch_loss = sess.run(loss, feed_dict={x: random_images, y: random_labels, keep_prob: dropout}) #keep_prob zmienione z 1. na dropout
            losses.append(batch_loss)
            
            stop_epoch = time()
            print("@@@ Epoch {} finished during {} - {}.".format(epoch, plot.time_format(start_epoch, 't'), plot.time_format(stop_epoch, 't')))
            print("    Network results:")
            print("    Itertion " + str(step) + ": batch loss = " + "{0:.5f}".format(batch_loss) + ", training accuracy = " + "{0:.5f}".format(acc))
            
            epoch += 1
            start_epoch = stop_epoch
        
        step += 1     
        
    stop_global = time()
    print("######################################################")
    print("Optimization finished. Started: {}, finished: {}, duration {}".format(plot.time_format(start_global, 'dt'), plot.time_format(stop_global, 'dt'), plot.time_interval(start_global, stop_global)))
    print("######################################################")
    
    test_size = min(400, train_images.shape[0])
    test_X = train_images[0:test_size,:]
    test_Y = train_labels[0:test_size,:]
    # Calculate accuracy 
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_X, y: test_Y, keep_prob: 1.}))