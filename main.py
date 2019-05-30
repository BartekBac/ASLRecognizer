import data_service as ds
import plot_service as plot
import tensorflow as tf
import cnn_service as cnn
from time import time
import numpy as np

count_of_single_letter = 30

train_images, train_labels = ds.load_train_data_fixed(limit_for_single_letter=count_of_single_letter)

# Parameters
learning_rate = 0.001
batch_size = 64
training_iters = 500 
display_step = 50
# Network Parameters
n_classes = 29
n_input = 64*64*3
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

##for image in train_images:
##    prediction = cnn.cnn_model(image)
##    print(prediction)

##random_images, random_labels = ds.get_random_batch(train_images, train_labels, 25)
##print('Drawn labels:')
##print(random_labels)

# Construct model
#pred = conv_net(x, weights, biases, keep_prob)
prediction = cnn.cnn_model(x, dropout)
# prepare loss and optimizer variables
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# evaluate model
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

######
# Initializing the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

losses = list()
accuracies = list()
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    step = 0
    epoch = 0
    start_epoch = time()
 
    # Keep training until reach max iterations
    while step <= training_iters:

        random_images, random_labels = ds.get_random_batch(train_images, train_labels, batch_size)
        
        # Fit training using batch data
        start_op = time()
        sess.run(optimizer, feed_dict={x: random_images, y: random_labels, keep_prob: dropout})
        end_op = time()
        print("#{} opt step {} {} takes {}".format(step,start_op,end_op, end_op-start_op))            
        
        if step % display_step == 0:
            
            print("acc start {}".format(time()))
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: random_images, y: random_labels, keep_prob: 1.})
            accuracies.append(acc)
            
            print("loss start {}".format(time()))
            # Calculate batch loss
            batch_loss = sess.run(loss, feed_dict={x: random_images, y: random_labels, keep_prob: 1.})
            losses.append(batch_loss)
            
            print("Iter " + str(step) + " started={}".format(time()) + ", Minibatch Loss= " + "{}".format(batch_loss) + ", Training Accuracy= " + "{}".format(acc))
            
            epoch += 1
        
        step += 1     
        
    end_epoch = time()
    print("Optimization Finished, end={} duration={}".format(end_epoch,end_epoch-start_epoch))
    
    
    test_size = min(400, train_images.shape[0])
    test_X = train_images[0:test_size,:]
    test_Y = train_labels[0:test_size,:]
    # Calculate accuracy 
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_X, y: test_Y, keep_prob: 1.}))