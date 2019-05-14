import data_service as ds
import plot_service as plot
import tensorflow as tf
import cnn_service as cnn

test_images, test_labels =  ds.load_test_data()
train_images, train_labels = ds.load_train_data(limit_for_single_letter=10)
#plot.plot_image_array(train_images, train_labels, columns_number=10)

for image in train_images:
    prediction = cnn.cnn_model(image)