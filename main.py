import data_service as ds
import plot_service as plot
import tensorflow as tf
import cnn_service as cnn

test_images, test_labels =  ds.load_test_data()
train_images, train_labels = ds.load_train_data(limit_for_single_letter=10)
#plot.plot_image_array(train_images, train_labels, columns_number=10)

filters = {
    'conv1': tf.random_normal([5,5,3,32]),  
    'conv2': tf.random_normal([5,5,32,64]) 
}

for image in train_images:    
    conv1 = cnn.conv2d(image, filter=filters['conv1'], strides=[1,1,1,1])
    pool1 = cnn.max_pool(conv1)
    conv2 = cnn.conv2d(pool1, filter=filters['conv2'], strides=[1,1,1,1])
    pool2 = cnn.max_pool(conv2)

