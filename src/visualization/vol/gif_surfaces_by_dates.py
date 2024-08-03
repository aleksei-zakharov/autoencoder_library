from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
import os
from IPython.display import Image, display


def gif_surfaces_by_dates(data,
                           dates,
                           x_labels,
                           y_labels
                           ):

    folder_path = '../../reports/vol/gifs/'

    max_vol = data.max()
    min_vol = data.min()
    x_space = np.arange(len(x_labels))
    y_space = np.arange(len(y_labels))
    X, Y = np.meshgrid(x_space, y_space)

    for i, date in enumerate(dates):
        
        surface = data[i]
        # Create plots
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')  # Create a 3D subplot    
        surf = ax.plot_surface(X, Y, surface, 
                               cmap=plt.get_cmap('Spectral_r'), 
                               linewidth=0, 
                               antialiased=False,
                               vmin=min_vol, 
                               vmax=max_vol)  # cmap='Spectral' # cmap='coolwarm'
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel('swap tenors')
        ax.set_xticks(ticks=x_space, labels=x_labels)
        ax.set_ylabel('option tenors') 
        ax.set_yticks(ticks=y_space, labels=y_labels)
        ax.set_zlabel('vol in bp')
        ax.set_zlim(min_vol, max_vol)
        date_string = f"{date.year}-{date.month:02}-{date.day:02}"
        ax.set_title(f'atm vol surface on ' + date_string)
        plt.savefig(folder_path + 'image_' + f"{i:04}" +'.png')  # save a plot as png image
        plt.close(fig)  # just not to show the graphs in python

    # Create an array of saved plots
    images = []
    for filename in sorted(glob.glob(folder_path + '*.png')):
        images.append(imageio.imread(filename))
    # Save a gif
    gif_file_path = folder_path + 'plot_surfaces_by_dates.gif'
    imageio.mimsave(gif_file_path, images, loop=0, fps=100)  # loop=0 means that gif will be played non-stop ; fps=100 is maximum

    # # Delete all created png files
    for filename in sorted(glob.glob(folder_path + '*.png')):
        os.remove(filename)

    # Open gif
    display(Image(gif_file_path))