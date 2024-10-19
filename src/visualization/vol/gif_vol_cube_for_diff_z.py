from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
import os
from IPython.display import Image, display
from PIL import Image as ImagePIL


def gif_vol_cube_for_diff_z(model,
                            normalizer,
                            all_z_vals,
                            z_idx,
                            x_labels,
                            y_labels,
                            strikes,
                            z_min=-3,
                            z_max=3,
                            z_points=31,
                            fps=3,
                            delete_pngs=True,
                            save_name=None):
    """
    Create a gif which display vol cube structure for different z0 / z1 values 

    
    Parameters:

    model: the variational autoencoder model

    normalizer: normalizer of vol cube data to [0,1] interval and vice versa (src.data.vol.normalizer file)

    all_z_vals: the list of z values based on which we construct volatility cube

    z_idx: index of z value that are changing to construct vol cube
    
    x_labels: list of labels of the x-axis on the grid graph (denotes swap tenors)

    y_labels: list of labels of the y-axis on the grid graph (denotes option tenors)

    uniq_strikes: list of strikes (for each strike, we construct a grid graph)

    strikes: list of all strikes in volatility data structure

    z_min / z_max: vol cubes are constructed for all latent space values z in interval [z_min, z_max]

    z_points: number of points between z_min and z_max for which vol cube will be constructed

    fps: the number of gif frames in 1 second (the higher, the faster gif goes)

    delete_pngs: if True, delete all created pngs files that were used to create a gif

    name: the name that is used here to name the saved gif. If it is not None, the plot is saved in the gif folder 
    """

    TICK_SIZE = 8
    SUBTITLE_SIZE = 10

    folder_path = '../../reports/vol/gifs/'
    z_space = np.linspace(z_min, z_max, z_points)


    # Find minimum and maximum vol values to have the same z ax for all plots with diverse z values
    max_vol = -float('inf')
    min_vol = float('inf')
    for idx, z in enumerate(z_space):
        # Change latent space value for certain z index
        all_z_vals[z_idx] = z
        # Make prediction
        norm_predictions = model.decoder.predict(np.array([all_z_vals]), verbose=0)[0]
        predictions = normalizer.denormalize(norm_predictions)
        max_vol = max(predictions.max(), max_vol)
        min_vol = min(predictions.min(), min_vol)     
        
    # Create values for x and y axes
    x_space = np.arange(len(x_labels))
    y_space = np.arange(len(y_labels))
    X, Y = np.meshgrid(x_space, y_space)

    # Create and save plots for different z values
    for idx, z in enumerate(z_space):
        # Change latent space value for certain z index
        all_z_vals[z_idx] = z

        # Calculate predictions via our model
        norm_predictions = model.decoder.predict(np.array([all_z_vals]), verbose=0)[0]
        predictions = normalizer.denormalize(norm_predictions)

        # Create plots
        fig = plt.figure(figsize=(16,3))
        fig.suptitle(f'vol cube data (in bp) for z{str(z_idx)} = {round(z,2)}')

        # Create subplots for each strike value
        for strk_idx, strk in enumerate(strikes):
            ax = fig.add_subplot(1, len(strikes), strk_idx + 1, projection='3d')  # Create a 3D subplot    
            surf = ax.plot_surface(X, Y, predictions[:, :, strk_idx], 
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
        file_path = folder_path + 'image_' + f"{idx:04}" +'.png'
        plt.savefig(file_path)  # save a plot as png image
        plt.close(fig)  # just not to show the graphs in python

        # Crop the image because plt.tight_layout() did not work
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
    if save_name is None:
        save_name = 'vol_cube_for_diff_z' + str(z_idx)
    imageio.mimsave(folder_path + save_name + '.gif', images, loop=0, fps=fps)  # loop=0 means that gif will be played non-stop
    
    # Delete all created png files
    if delete_pngs:
        for filename in sorted(glob.glob(folder_path + '*.png')):
            os.remove(filename)

    # Open gif
    display(Image(folder_path + save_name + '.gif'))