import os
import cv2
import random
from time import time
from math import floor
import tensorflow as tf
import numpy as np
root_data_dir = "C:/rep/VI sem/BIAI/data"
#root_data_dir = '/home/martyna/Studia/biai'
test_data_dir = root_data_dir + '/asl_alphabet_test'
train_data_dir = root_data_dir + '/asl_alphabet_train'
classes_count = 29
letters_dictionary = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,
                      'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,
                      'V':21,'W':22,'X':23,'Y':24,'Z':25,'space':26,'del':27,'nothing':28}

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
    return np.array(images_to_return), np.array(labels_to_return)

def load_train_data_fixed(limit_for_single_letter=3000):
    if limit_for_single_letter > 3000 or limit_for_single_letter < 1:
        print("Limit to load train data is in range <1; 3000> per letter.")
        return [], []

    destination_image_size = 64,64
    color_channel_count = 3
    images_count = limit_for_single_letter * classes_count
    single_image_size = destination_image_size[0] * destination_image_size [1] * color_channel_count
    images_to_return = np.zeros([images_count,single_image_size])
    labels_to_return = np.zeros([images_count,classes_count])
    i = 0
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
            #image = tf.cast(image, tf.float32)
            #image = tf.reshape(image, shape=[-1, destination_image_size[0], destination_image_size[1], color_channel_count])
            images_to_return[i,:] = image.flatten()
            labels_to_return[i,:] = letter2vector(letter)
            i += 1
        local_stop = time()
        print(' in ' + "{0:.3f}".format(local_stop - local_start) + ' sec.')
    global_stop = time()
    print('=======================================')
    print('=======================================')
    print('Fetched ' + str(29 * limit_for_single_letter) + ' files in ' +  "{0:.3f}".format(global_stop - global_start) + ' sec.')
    return images_to_return, labels_to_return

def get_random_batch(images, labels, batch_size):
    images_count = len(images)
    single_image_size = 64 * 64 * 3
    random_indexes = []
    images_to_return = np.zeros([images_count,single_image_size])
    labels_to_return = np.zeros([images_count,classes_count])
    try:
        random_indexes = random.sample(range(0, images_count-1), batch_size)
        for i in range(batch_size):
            images_to_return[i,:] = images[random_indexes[i]]
            labels_to_return[i,:] = labels[random_indexes[i]]
    except ValueError:
        print('Batch size exceeded population size.')

    return images_to_return, labels_to_return

def letter2vector(letter):
    vector = np.zeros(classes_count)
    vector[letters_dictionary[letter]] = 1
    return vector

