import tensorflow as tf

def conv2d(inputs, filter, strides, padding='SAME'):
    return tf.nn.relu(
                tf.nn.conv2d(
                    input = inputs, 
                    filter = filter, 
                    strides = strides, 
                padding = padding
                )
            )
                   
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME'):
    return tf.nn.max_pool(
            value = inputs,
            ksize = ksize,
            strides = strides,
            padding = padding
        )
    
