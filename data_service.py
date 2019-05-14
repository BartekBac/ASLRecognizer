import os
import cv2
from time import time
from math import floor
import tensorflow as tf
root_data_dir = '/home/martyna/Studia/biai'
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
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, shape=[-1, 64,64,3])
        label = file
        label = label[:-9]
        images_to_return.append(image)
        labels_to_return.append(label)
    return images_to_return, labels_to_return

def load_train_data(limit_for_single_letter=3000):
    if limit_for_single_letter > 3000 or limit_for_single_letter < 1:
        print("Limit to load train data is in range <1; 3000> per letter.")
        return [], []

    destination_image_size = 64,64 
    images_to_return = []
    labels_to_return = []
    print_bar_limit = 20
    global_start = time()
    for letter in os.listdir(train_data_dir):
        print('Fetching ' + str(limit_for_single_letter) + ' files for "' + str(letter) + '"', end='')  
        print_step = floor(limit_for_single_letter / 20)
        current_step = print_step
        if limit_for_single_letter > print_bar_limit:
            print('\n0%                100%') 
        local_start = time()
        for file_index in range(1, (limit_for_single_letter+1)):
            if limit_for_single_letter > print_bar_limit and (file_index - 1) >= current_step:
                print('=', end='', flush=True)
                current_step += print_step
            file_path = train_data_dir + '/' + str(letter) + '/' + str(letter) + str(file_index) + '.jpg'
            image = cv2.imread(file_path)
            image = cv2.resize(image, destination_image_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, shape=[-1, 64,64,3])
            label = letter
            images_to_return.append(image)
            labels_to_return.append(label)
        local_stop = time()
        print(' in ' + "{0:.3f}".format(local_stop - local_start) + ' sec.')
    global_stop = time()
    print('=======================================')
    print('=======================================')
    print('Fetched ' + str(29 * limit_for_single_letter) + ' files in ' +  "{0:.3f}".format(global_stop - global_start) + ' sec.')
    return images_to_return, labels_to_return