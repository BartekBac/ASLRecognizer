import data_service as ds
import plot_data as plot
test_images, test_labels =  ds.load_test_data()
print('Labels:')
print(test_labels)
plot.plot_image_array(test_images, test_labels, figure_size=(10,10))