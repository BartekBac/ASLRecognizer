import matplotlib.pyplot as plt
from math import ceil, floor

def plot_single_image(image, label):
    plt.axis('off')
    plt.imshow(image)
    plt.title(label)
    return

def plot_image_array(image_array, label_array, columns_number=4, figure_size=(15,15)):
    images_count = len(image_array)
    if images_count > 300:
        print("Cannot plot more than 300 images.")
        return
    rows_number = ceil(images_count/columns_number)
    fig = plt.figure(figsize = figure_size)
    gs = fig.add_gridspec(rows_number, columns_number)
    gs.update(hspace = 0.5, wspace=0.05)
    print("Plotting " + str(images_count) + " images...")
    print('0%                100%')
    print_step = floor(images_count / 20)
    current_step = print_step
    for i in range(images_count):
        if i >= current_step:
            print('=', end='', flush=True)
            current_step += print_step 
        yi = floor(i/float(columns_number))
        xi = i % columns_number
        fig.add_subplot(gs[yi, xi])
        plot_single_image(image_array[i], label_array[i])
    print('\nPlotted ' + str(images_count) + ' images in ' + str(rows_number) + ' rows / ' + str(columns_number) + ' columns.')
    plt.show()