import matplotlib.pyplot as plt
from matplotlib import cm  # to change color scheme


def plot_latent_space_2d_values(model, x, y=None):
    # Because our latent space is two-dimensional, there are a few cool visualizations that can be done at this point. One is to look at the neighborhoods of different classes on the latent 2D plane
    if model.model_type == 'vae':
        x_encoded, __, __ = model.encoder.predict(x, verbose=0)  # z_mean, z_var, z
    elif model.model_type == 'ae':
        x_encoded = model.encoder.predict(x, verbose=0)  # z
    plt.figure(figsize=(12, 6))
    plt.xlabel('first latent space variable')
    plt.ylabel('second latent space variable')
    plt.title('Latent space distributions for each of 10 digits of real data')
    if y is None:
        plt.scatter(x_encoded[:,0], x_encoded[:,1], s=1)  # s=2 initially
    else:
        plt.scatter(x_encoded[:,0], x_encoded[:,1], c=y, s=1, cmap=cm.rainbow)  # s=2 initially
    plt.colorbar()
    # # save plot
    # if save_name is not None:
    #     # make a folder
    #     folder_path = os.path.join('../reports', save_name)
    #     os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
    #     # save plot if the file doesn't exist
    #     file_path = os.path.join('../reports', save_name, 'predictions_of_latent_space.png')
    #     if os.path.exists(file_path):
    #         raise FileExistsError(f"The file '{file_path}' already exists.")
    #     else:
    #         plt.savefig(file_path)
    # display plot
    plt.show()