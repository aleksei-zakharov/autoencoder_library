import os
import matplotlib.pyplot as plt
import numpy as np


def plot_true_vs_prediction_images(model, 
                                   x,
                                   y,
                                   save_name=None):
    """
    Draw N samples of comparison between original images of handwritten digits and reconstructed images for each of 0, 1, ..., 9 digits

    
    Parameters:

    model: the model such as autoencoder (keras.Model), variational autoencoder (keras.Model) or PCA method
    
    x: the inputs of the model (mnist dataset of handwritten-digit images)

    y: the labels of input data. It is used only for mnist dataset: the real digits such as 0, 1, ..., 9.

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    N = 5  # number of samples to be shown for each value 0, 1, 2, ... 9

    plt.figure(figsize=(13, 3.2))
    for n_col in range(10):  # digits 0, 1, 2, ..., 9
        idx = 0 # index of current element in x_train
        for n_row in range(N):  # samples
            # Find next picture showing n_row value (starting from idx index)
            while y[idx] != n_col:
                idx += 1
            # Display original picture
            ax = plt.subplot(N, 20, n_row*20 + n_col*2 + 1)  # number of rows, number of columns, graph index
            plt.imshow(x[idx].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display picture deconstred via autoencoder
            if model.model_type in ['vae','ae']:
                decoded_img = model.predict(x[idx][None,...], verbose=0) # to predict only 1 element. The result has (1,784) shape 
            elif model.model_type == 'pca':
                # project the dataset onto the principal components
                pca_comp = model.transform(np.expand_dims(x[idx],0))
                # transform the projections back into the input space
                pca_recon = model.inverse_transform(pca_comp)
                decoded_img = pca_recon  # to predict only 1 element. The result has (1,784) shape 
            ax = plt.subplot(N, 20, n_row*20 + n_col*2 + 2)  # number of rows, number of columns, graph index
            plt.imshow(decoded_img.reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            idx += 1 # to find the next index whos picture shows n_row value

    # Save plot
    if save_name is not None:
        # Make a folder if it doesn't exist
        folder_path = '../../reports/mnist'
        os.makedirs(folder_path, exist_ok=True)
        # Save plot
        file_path = os.path.join(folder_path, save_name + '_true_vs_prediction_images.png')
        plt.savefig(file_path)
            
    # Display plot
    plt.show()