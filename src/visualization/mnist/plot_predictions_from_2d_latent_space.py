import os
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_from_2d_latent_space(model, 
                                          n_cols, 
                                          xmin, 
                                          xmax, 
                                          ymin, 
                                          ymax,
                                          save_name=None):
    # Model can be ae or vae
    latent_space_x = np.linspace(xmin, xmax, n_cols)
    latent_space_y = np.linspace(ymax, ymin, n_cols)
    
    plt.figure(figsize=(int(n_cols / 2), int(n_cols / 2)))

    for n_row in range(n_cols):
        for n_col in range(n_cols):
            ax = plt.subplot(n_cols, n_cols, n_cols * n_row + n_col + 1)
            x = latent_space_x[n_col]
            y = latent_space_y[n_row]
            img = model.decoder.predict(np.array([[x, y]]), verbose=0)
            plt.imshow(img.reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # Save plot
    if save_name is not None:
        # Make a folder
        folder_path = '../../reports/mnist'
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, save_name + '_pred_from_lat_space.png')
        plt.savefig(file_path)

    # Display plot
    plt.show()