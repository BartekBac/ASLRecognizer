import matplotlib.pyplot as plt
from math import ceil, floor

def plot_single_image(image, label):
    plt.axis('off')
    plt.imshow(image)
    plt.title(label)
    return

def plot_image_array(image_array, label_array, columns_number=4, figure_size=(15,15)):
    images_count = len(image_array)
    rows_number = ceil(images_count/columns_number)
    fig = plt.figure(figsize = figure_size)
    gs = fig.add_gridspec(rows_number, columns_number)
    gs.update(hspace = 0.5, wspace=0.05)
    for i in range(images_count):
        yi = floor(i/float(columns_number))
        xi = i % columns_number
        fig.add_subplot(gs[yi, xi])
        plot_single_image(image_array[i], label_array[i])
    print('Plotted ' + str(images_count) + ' images in ' + str(rows_number) + ' rows / ' + str(columns_number) + ' columns.')
    plt.show()