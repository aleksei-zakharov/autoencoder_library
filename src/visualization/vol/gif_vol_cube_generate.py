from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
import os
from IPython.display import Image, display
from PIL import Image as ImagePIL


def gif_vol_cube_generate(model,
                          normalizer,
                          x_labels,
                          y_labels,
                          uniq_strikes,
                          strikes,
                          z_min=-4,
                          z_max=4,
                          fps=3,
                          delete_pngs=True,
                          name=None):
    """
    Draw latent space scatter plot of latent space variables calculated from inputs

    
    Parameters:

    model: the variational autoencoder model

    normalizer: normalizer of vol cube data to [0,1] interval and vice versa (src.data.vol.normalizer file)
    
    x_labels: list of labels of the x-axis on the grid graph (denotes swap tenors)

    y_labels: list of labels of the y-axis on the grid graph (denotes option tenors)

    uniq_strikes: list of strikes (for each strike, we construct a grid graph)

    strikes: list of all strikes in volatility data structure

    z_min / z_max: vol cubes are generated for all latent space values z in interval [z_min, z_max]

    fps: the number of gif frames in 1 second (the higher, the faster gif goes)

    delete_pngs: if True, delete all created pngs files that were used to create a gif

    vae_latent_type: type of latent space variable that must be plotted. Possible values: 'z', 'z_mean', 'z_logvar'.

    name: the name that is used here to name the saved gif. If it is not None, the plot is saved in the gif folder 
    """

    TICK_SIZE = 8
    SUBTITLE_SIZE = 10

    folder_path = '../../reports/vol/gifs/'
    z_interval = np.linspace(z_min, z_max, z_max - z_min + 1)

    # Find minimum and maximum vol values to have the same z ax for all plots with diverse z values
    max_vol = -float('inf')
    min_vol = float('inf')
    for i_z0, z0 in enumerate(z_interval):
        for i_z1, z1 in enumerate(z_interval):
            # Make prediction
            norm_predictions = model.decoder.predict(np.array([[z0, z1]]), verbose=0)[0]
            predictions = normalizer.denormalize(norm_predictions)
            max_vol = max(predictions.max(), max_vol)
            min_vol = min(predictions.min(), min_vol)     
        
    # Create values for x and y axes
    x_space = np.arange(len(x_labels))
    y_space = np.arange(len(y_labels))
    X, Y = np.meshgrid(x_space, y_space)

    # Create and save plots for different z values
    for i_z0, z0 in enumerate(z_interval):
        for i_z1, z1 in enumerate(z_interval):
            # Calculate predictions via our model
            norm_predictions = model.decoder.predict(np.array([[z0, z1]]), verbose=0)[0]
            predictions = normalizer.denormalize(norm_predictions)

            # Create plots
            fig = plt.figure(figsize=(16,3))
            fig.suptitle(f'vol cube data (in bp) for z0 = {round(z0,2)} and z1 = {round(z1,2)}')

            # Create subplots for each strike value
            for strk_idx, strk in enumerate(strikes):
                ax = fig.add_subplot(1, len(strikes), strk_idx + 1, projection='3d')  # Create a 3D subplot    
                surf = ax.plot_surface(X, Y, predictions[:, :, uniq_strikes.index(strk)], 
                                        cmap=plt.get_cmap('Spectral_r'), 
                                        linewidth=0, 
                                        antialiased=False, 
                                        vmin=min_vol, 
                                        vmax=max_vol)

                # ax.view_init(elev=25, azim=315)  # change the viewpoint
                ax.set_xlabel('swap tenors', fontsize = TICK_SIZE)
                ax.set_xticks(ticks=x_space, labels=x_labels, size=TICK_SIZE)
                ax.set_ylabel('option tenors', fontsize = TICK_SIZE) 
                ax.set_yticks(ticks=y_space, labels=y_labels, size=TICK_SIZE)
                ax.set_zlim(min_vol, max_vol)
                ax.tick_params(axis='z', labelsize=TICK_SIZE)
                ax.set_title(f'strike: {strk}', size=SUBTITLE_SIZE)

            # Save the image
            idx = i_z0 * len(z_interval) + i_z1
            file_path = folder_path + 'image_' + f"{idx:04}" +'.png'
            plt.savefig(file_path)  # save a plot as png image
            plt.close(fig)  # just not to show the graphs in python

            # Crop the image
            img = ImagePIL.open(file_path)
            width, height = img.size
            box = (int(0.11 * width), 0, int(0.93 * width), height)  # left, top, right, bottom
            img = img.crop(box)
            img.save(file_path)

    # Create an array of saved plots
    images = []
    for filename in sorted(glob.glob(folder_path + '*.png')):
        images.append(imageio.imread(filename))
        
    # Save a gif
    if name is None:
        name = 'vol_cube_generate'
    imageio.mimsave(folder_path + name + '.gif', images, loop=0, fps=fps)  # loop=0 means that gif will be played non-stop
    
    # Delete all created png files
    if delete_pngs:
        for filename in sorted(glob.glob(folder_path + '*.png')):
            os.remove(filename)

    # Open gif
    display(Image(folder_path + name + '.gif'))