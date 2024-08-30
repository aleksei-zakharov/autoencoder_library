import os
import matplotlib.pyplot as plt


def plot_true_images(N,
                     x,
                     y,
                     save_name=None):
    # Graph shows comparison between original pictures and predicted pictures
    # N = 5  # number of samples to be shown for each value 0, 1, 2, ... 9

    plt.figure(figsize=(5, 1.7))
    for n_col in range(10):  # digits 0, 1, 2, ..., 9
        idx = 0 # index of current element in X_train
        for n_row in range(N):  # samples
            # Find next picture showing n_row value (starting from idx index)
            while y[idx] != n_col:
                idx += 1
            # Display original picture
            ax = plt.subplot(N, 10, n_row*10 + n_col + 1)  # number of rows, number of columns, graph index
            plt.imshow(x[idx].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            idx += 1 # to find the next index whos picture shows n_row value

    # Save plot
    if save_name is not None:
        # Make a folder
        folder_path = '../../reports/mnist'
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, save_name + '_real_images.png')
        plt.savefig(file_path)
            
    # Display plot
    plt.show()