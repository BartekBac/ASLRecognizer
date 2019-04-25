import os
import cv2
import matplotlib.pyplot as plt

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

test_images, test_labels = load_test_data()
print('Labels:')
print(test_labels)

fig = plt.figure(figsize = (15,15))
def plot_image(fig, image, label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image)
    plt.title(label)
    return

image_index = 0
row = 5
col = 6
for i in range(1,(row*col)-1):
    plot_image(fig, test_images[image_index], test_labels[image_index], row, col, i)
    image_index = image_index + 1
plt.show()
