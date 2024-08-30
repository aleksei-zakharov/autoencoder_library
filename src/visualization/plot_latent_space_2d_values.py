import os
import matplotlib.pyplot as plt
from matplotlib import cm  # to change color scheme


def plot_latent_space_2d_values(model, 
                                x,
                                y=None,
                                vae_latent_type=None,
                                data_type='mnist',  # type = 'mnist' or 'vol'
                                save_name=None):
    # Because our latent space is two-dimensional, there are a few cool visualizations that can be done at this point. One is to look at the neighborhoods of different classes on the latent 2D plane
    if vae_latent_type is None:
        vae_latent_type = 'z'

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
        
    plt.figure(figsize=(8, 6))
    plt.xlabel('first latent space variable (z0)')
    plt.ylabel('second latent space variable (z1)')
    plt.title('Latent space (' + vae_latent_type + ' values) encoded based on the NN inputs')
    if y is None:
        plt.scatter(x_encoded[:,0], x_encoded[:,1], s=1)  # s=2 initially
    else:
        plt.scatter(x_encoded[:,0], x_encoded[:,1], c=y, s=1, cmap=cm.rainbow)  # s=2 initially
        plt.colorbar()

    # Save plot
    if save_name is not None:
        # Make a folder
        folder_path = '../../reports/' + data_type
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, save_name + '_2d_' + vae_latent_type + '.png')
        plt.savefig(file_path)

    # Display plot
    plt.show()