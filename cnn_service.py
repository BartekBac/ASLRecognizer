import tensorflow as tf

def conv2d(inputs, bias, filter, strides, padding='SAME'):
  return tf.nn.relu( 
                tf.nn.bias_add( 
                    tf.nn.conv2d(
                        input = inputs, 
                        filter = filter, 
                        strides = strides, 
                        padding = padding),
                        bias))   
                   
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'): #strides zmienione z [1,1,1,1] na [1,2,2,1]
    return tf.nn.max_pool(
            value = inputs,
            ksize = ksize,
            strides = strides,
            padding = padding
            )


def cnn_model(input_layer, param, dropout=0.75):

    input_layer = tf.reshape(input_layer, shape=[-1, 64,64,3])

    conv1 = conv2d(input_layer, param['bias_c1'], filter=param['filter_c1'], strides=[1,1,1,1])
    pool1 = max_pool(conv1)
    
    conv2 = conv2d(pool1, param['bias_c2'], filter=param['filter_c2'], strides=[1,1,1,1])
    pool2 = max_pool(conv2)

    conv3 = conv2d(pool2, param['bias_c3'], filter = param['filter_c3'], strides=[1,1,1,1])
    pool3 = max_pool(conv3)

    fully_c = tf.reshape(pool3, [-1,8*8*64])
    fully_c = tf.nn.relu(
                tf.add(
                    tf.matmul(fully_c, param['weights_f']), 
                    param['bias_f'],
            ))  

    fully_c = tf.nn.dropout(fully_c, rate=dropout)

    logits = tf.add(
                tf.matmul(fully_c, param['weights_l']),
                param['bias_l']
            )

    return logits