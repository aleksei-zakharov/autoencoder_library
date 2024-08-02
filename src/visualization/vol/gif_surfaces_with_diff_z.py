from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
import os
from IPython.display import Image, display


def gif_surfaces_with_diff_z(model,
                            normalizer,
                            all_z_vals,
                            z_idx,
                            x_labels,
                            y_labels,
                            z_min=-3,
                            z_max=3,
                            z_points=31,
                            fps=5,
                            rot_angle=0, 
                            name=None):
    
    folder_path = '../../reports/vol/gifs/'
    z_space = np.linspace(z_min, z_max, z_points)


    # Find minimum and maximum vol values to have the same z ax for all plots with diverse z values
    max_vol = -float('inf')
    min_vol = float('inf')
    for idx, z in enumerate(z_space):
        # Rotation of z0 and z1 vectors
        if z_idx == 0:
            all_z_vals[0] = z * np.cos(rot_angle / 180 * np.pi)
            all_z_vals[1] = z * np.sin(rot_angle / 180 * np.pi)
        else:
            all_z_vals[0] = z * np.sin(rot_angle / 180 * np.pi)
            all_z_vals[1] = - z * np.cos(rot_angle / 180 * np.pi)

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
        # Calculate predictions via our model

        # Rotation of z0 and z1 vectors
        if z_idx == 0:
            all_z_vals[0] = z * np.cos(rot_angle / 180 * np.pi)
            all_z_vals[1] = z * np.sin(rot_angle / 180 * np.pi)
        else:
            all_z_vals[0] = z * np.sin(rot_angle / 180 * np.pi)
            all_z_vals[1] = - z * np.cos(rot_angle / 180 * np.pi)

        norm_predictions = model.decoder.predict(np.array([all_z_vals]), verbose=0)[0]
        predictions = normalizer.denormalize(norm_predictions)

        # Create plots
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')  # Create a 3D subplot
        surf = ax.plot_surface(X, Y, predictions, cmap=plt.get_cmap('Spectral_r'), linewidth=0, antialiased=False)  # cmap='Spectral' # cmap='coolwarm'
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel('swap tenors')
        ax.set_xticks(ticks=x_space, labels=x_labels)
        ax.set_ylabel('option tenors') 
        ax.set_yticks(ticks=y_space, labels=y_labels)
        ax.set_zlabel('vol in bp')
        ax.set_zlim(min_vol, max_vol)
        ax.set_title(f'vol surface for z{str(z_idx)} = {round(z,2)}')
        plt.savefig(folder_path + 'image_' + f"{idx:03}" +'.png')  # save a plot as png image
        plt.close(fig)  # just not to show the graphs in python

    # Create an array of saved plots
    images = []
    for filename in sorted(glob.glob(folder_path + '*.png')):
        images.append(imageio.imread(filename))
    # Save a gif
    if name is None:
        name = 'plot_surface_for_diff_z' + str(z_idx)
    imageio.mimsave(folder_path + name + '.gif', images, loop=0, fps=fps)  # loop=0 means that gif will be played non-stop
    # Delete all created png files
    for filename in sorted(glob.glob(folder_path + '*.png')):
        os.remove(filename)
    # Open gif
    display(Image(folder_path + name + '.gif'))