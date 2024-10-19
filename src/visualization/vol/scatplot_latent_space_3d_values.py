import os
import matplotlib.pyplot as plt
from matplotlib import cm  # to change color scheme
import matplotlib.pyplot as plt
import numpy as np


def scatplot_latent_space_3d_values(model, 
                                    label1,
                                    data1=None,
                                    label2=None,
                                    data2=None,
                                    gaussian_num=1000,
                                    save_name=None):
    """
    Draw latent space scatter plot of latent space variables calculated from inputs

    Parameters:

    model: the model such as autoencoder (keras.Model), variational autoencoder (keras.Model) or PCA method
    
    data1: the inputs of the model

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    # Create latent space variables for data1 dataset
    if label1 == 'train data' or label1 == 'test data':
        if model.model_type == 'vae':  
                __, __, z_vals1 = model.encoder.predict(data1, verbose=0)  # z_mean, z_logvar, z
        elif model.model_type == 'ae':
            z_vals1 = model.encoder.predict(data1, verbose=0)  # z
        elif model.model_type == 'pca':
            z_vals1 = model.transform(data1)  # project the dataset onto the principal components
    elif label1 == 'gaussian':
        np.random.seed(0)
        z_vals1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), gaussian_num)

    # Create latent space variables for data2 dataset
    if label2 is not None:
        if label2 == 'train data' or label2 == 'test data':
            if model.model_type == 'vae':  
                    __, __, z_vals2 = model.encoder.predict(data2, verbose=0)  # z_mean, z_logvar, z
            elif model.model_type == 'ae':
                z_vals2 = model.encoder.predict(data2, verbose=0)  # z
            elif model.model_type == 'pca':
                z_vals2 = model.transform(data2)  # project the dataset onto the principal components
        elif label2 == 'gaussian':
            np.random.seed(0)
            z_vals2 = np.random.multivariate_normal(np.zeros(3), np.eye(3), gaussian_num)


    fig, axs = plt.subplots(1,3, figsize=(12, 4))
    plt.suptitle('Latent space variable values encoded based on the NN inputs')
    axs = axs.ravel()

    # Create three 2-dimensional plots
    INDEXES = [[0, 1], [0, 2], [1, 2]]
    for i, indexes in enumerate(INDEXES):
        idx1, idx2 = indexes
        axs[i].set_xlabel(f'latent space variable z{idx1}')
        axs[i].set_ylabel(f'latent space variable z{idx2}')
        axs[i].scatter(z_vals1[:,idx1], z_vals1[:,idx2], s=1, label=label1, color='orange')  # s=2 initially
        axs[i].scatter(z_vals2[:,idx1], z_vals2[:,idx2], s=1, label=label2, color='tab:blue')  # s=2 initially
        axs[i].legend()
    plt.tight_layout()

    # Save plot
    if save_name is not None:
        # Make a folder if it doesn't exist
        folder_path = '../../reports/vols/'
        # Save plot
        file_path = os.path.join(folder_path, save_name + '_3d_latent_space.png')
        plt.savefig(file_path)

    # Display plot
    plt.show()