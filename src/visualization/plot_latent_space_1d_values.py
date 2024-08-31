import matplotlib.pyplot as plt
from matplotlib import cm  # to change color scheme
import numpy as np
import os


def plot_latent_space_1d_values(model, 
                                x, 
                                data_type,
                                y=None,
                                vae_latent_type=None,
                                save_name=None):
    """
    Draw the histogram of latent space variables calculated from inputs

    
    Parameters:

    model: the model such as autoencoder (keras.Model), variational autoencoder (keras.Model) or PCA method
    
    x: the inputs of the model (mnist dataset of handwritten-digit images or volatility cube data)
    
    data_type: type of data to be plotted. Possible values: 'mnist' or 'vol'. The plot is saved in the folder with this name

    y: the labels of input data. It is used only for mnist dataset: the real digits such as 0, 1, ..., 9.

    vae_latent_type: type of latent space variable that must be plotted. Possible values: 'z', 'z_mean', 'z_logvar'.

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    if vae_latent_type is None:
        vae_latent_type = 'z'

    # The calculation of latent space variables for different model types
    if model.model_type == 'vae':
        if vae_latent_type == 'z_mean':
            x_encoded, __, __ = model.encoder.predict(x, verbose=0)  # z_mean, z_logvar, z
        elif vae_latent_type == 'z_logvar':
            __, x_encoded, __ = model.encoder.predict(x, verbose=0)  # z_mean, z_logvar, z
        elif vae_latent_type == 'z':
            __, __, x_encoded = model.encoder.predict(x, verbose=0)  # z_mean, z_logvar, z
    elif model.model_type == 'ae':
        x_encoded = model.encoder.predict(x, verbose=0)  # z

    # Histogram of the latent space variable
    bins = np.linspace(np.min(x_encoded), np.max(x_encoded), 400)
    if y is None:
        plt.hist(x_encoded, bins, alpha=0.5)
    else:
        for i in range(10):
            idxs = [key for key, val in enumerate(y) if val==i]
            plt.hist(x_encoded[idxs], bins, alpha=0.5, label=i)
        plt.legend()
    plt.title('Latent space (' + vae_latent_type + ') distribution of real data')

    # Save plot
    if save_name is not None:
        # Make a folder if it doesn't exist
        folder_path = '../../reports/' + data_type
        os.makedirs(folder_path, exist_ok=True)
        # Save plot
        file_path = os.path.join(folder_path, save_name + '_1d_' + vae_latent_type + '.png')
        plt.savefig(file_path)

    # Display plot
    plt.show()