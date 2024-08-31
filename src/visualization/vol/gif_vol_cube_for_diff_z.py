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
                            uniq_strikes,
                            strikes,
                            z_min=-3,
                            z_max=3,
                            z_points=31,
                            fps=3,
                            delete_pngs=True,
                            name=None):
    

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
            surf = ax.plot_surface(X, Y, predictions[:, :, uniq_strikes.index(strk)], 
                                    cmap=plt.get_cmap('Spectral_r'), 
                                    linewidth=0, 
                                    antialiased=False, 
                                    vmin=min_vol, 
                                    vmax=max_vol)  # cmap='Spectral' # cmap='coolwarm'       

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
        name = 'vol_cube_for_diff_z' + str(z_idx)
    imageio.mimsave(folder_path + name + '.gif', images, loop=0, fps=fps)  # loop=0 means that gif will be played non-stop
    
    # Delete all created png files
    if delete_pngs:
        for filename in sorted(glob.glob(folder_path + '*.png')):
            os.remove(filename)

    # Open gif
    display(Image(folder_path + name + '.gif'))