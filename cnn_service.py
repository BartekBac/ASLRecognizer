import tensorflow as tf

def conv2d(inputs, filter, strides, padding='SAME'):
    return tf.nn.relu(
                tf.nn.conv2d(
                    input = inputs, 
                    filter = filter, 
                    strides = strides, 
                    padding = padding
            ))
                   
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME'):
    return tf.nn.max_pool(
            value = inputs,
            ksize = ksize,
            strides = strides,
            padding = padding
            )

def cnn_model(input_layer, dropout=0.75):
    filter_c1 = tf.random_normal([5,5,3,32]) 
    conv1 = conv2d(input_layer, filter=filter_c1, strides=[1,1,1,1])
    pool1 = max_pool(conv1)
    
    filter_c2 = tf.random_normal([5,5,32,64])
    conv2 = conv2d(pool1, filter=filter_c2, strides=[1,1,1,1])
    pool2 = max_pool(conv2)

    weights_f = tf.random_normal([16*16*64, 1024]) 
    fully_c = tf.reshape(pool2, [-1,16*16*64])
    fully_c = tf.nn.relu(
                tf.add(
                    tf.matmul(fully_c, weights_f), 
                    tf.random_normal([1024])
                ))  

    fully_c = tf.nn.dropout(fully_c, rate=dropout)

    classes_quantity = 29
    weights_l = tf.random_normal([1024, classes_quantity ])
    logits = tf.add(
                tf.matmul(fully_c, weights_l),
                tf.random_normal([classes_quantity]) 
                )

    predictions = tf.nn.softmax(logits)
    return predictions