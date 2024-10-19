import os
import matplotlib.pyplot as plt
from matplotlib import cm  # to change color scheme


def scatplot_latent_space_2d_values(model, 
                                x,
                                y=None,
                                indexes=[0,1],
                                vae_latent_type='z',
                                data_type=None,
                                save_name=None):
    """
    Draw latent space scatter plot of latent space variables calculated from inputs

    
    Parameters:

    model: the model such as autoencoder (keras.Model), variational autoencoder (keras.Model) or PCA method
    
    x: the inputs of the model (mnist dataset of handwritten-digit images or volatility cube data)

    y: the labels of input data. It is used only for mnist dataset: the real digits such as 0, 1, ..., 9.

    vae_latent_type: type of latent space variable that must be plotted. Possible values: 'z', 'z_mean', 'z_logvar'.

    data_type: type of data to be plotted. Possible values: 'mnist' or 'vol'. The plot is saved in the folder with this name

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

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
    elif model.model_type == 'pca':
        x_encoded = model.transform(x)  # project the dataset onto the principal components

    # Plot the latent space variables
    plt.figure(figsize=(5, 5))
    plt.xlabel(f'latent space variable z{indexes[0]}')
    plt.ylabel(f'latent space variable z{indexes[1]}')
    plt.title('Latent space (' + vae_latent_type + ' values) encoded based on the NN inputs')
    if y is None:
        plt.scatter(x_encoded[:,indexes[0]], x_encoded[:,indexes[1]], s=1)  # s=2 initially
    else:
        plt.scatter(x_encoded[:,indexes[0]], x_encoded[:,indexes[1]], c=y, s=1, cmap=cm.rainbow)  # s=2 initially
        plt.colorbar()
    plt.tight_layout()
    
    # Save plot
    if save_name is not None:
        if data_type is None:
            raise NameError('data_type parameter was not defined (possible values are \'mnist\' or \'vol\')')
        else:
            # Make a folder if it doesn't exist
            folder_path = '../../reports/' + data_type
            os.makedirs(folder_path, exist_ok=True)
            # Save plot
            file_path = os.path.join(folder_path, save_name + '_2d_' + vae_latent_type + '.png')
            plt.savefig(file_path)

    # Display plot
    plt.show()