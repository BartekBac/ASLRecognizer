import os
import cv2

root_data_dir = 'c:/rep/VI sem/BIAI/data'
test_data_dir = root_data_dir + '/asl_alphabet_test'
train_data_dir = root_data_dir + '/asl_alphabet_train'
def load_test_data():
    destination_image_size = 64,64 
    images_to_return = []
    labels_to_return = []
    for file in os.listdir(test_data_dir):
        file_path = test_data_dir + '/' + file
        image = cv2.imread(file_path)
        image = cv2.resize(image, destination_image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = file
        label = label[:-9]
        images_to_return.append(image)
        labels_to_return.append(label)
    return images_to_return, labels_to_return
    