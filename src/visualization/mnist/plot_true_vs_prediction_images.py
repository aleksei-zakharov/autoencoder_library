import os
import matplotlib.pyplot as plt


def plot_true_vs_prediction_images(model, x, y, save_name=None):
    # graph shows comparison between original pictures and predicted pictures
    N = 5  # number of samples to be shown for each value 0, 1, 2, ... 9

    plt.figure(figsize=(16, 5))
    for n_col in range(10):  # digits 0, 1, 2, ..., 9
        idx = 0 # index of current element in X_train
        for n_row in range(N):  # samples
            # find next picture showing n_row value (starting from idx index)
            while y[idx] != n_col:
                idx += 1
            # display original picture
            ax = plt.subplot(N, 20, n_row*20 + n_col*2 + 1)  # number of rows, number of columns, graph index
            plt.imshow(x[idx].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display picture deconstred via autoencoder
            decoded_img = model.predict(x[idx][None,...], verbose=0) # to predict only 1 element. The result has (1,784) shape 
            ax = plt.subplot(N, 20, n_row*20 + n_col*2 + 2)  # number of rows, number of columns, graph index
            plt.imshow(decoded_img.reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            idx += 1 # to find the next index whos picture shows n_row value
    # save plot
    if save_name is not None:
        # make a folder
        folder_path = os.path.join('../reports', save_name)
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # save plot if the file doesn't exist
        file_path = os.path.join('../reports', save_name, 'predictions.png')
        if os.path.exists(file_path):
            raise FileExistsError(f"The file '{file_path}' already exists.")
        else:
            plt.savefig(file_path)
    # display plot
    plt.show()