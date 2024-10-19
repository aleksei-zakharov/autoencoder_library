from matplotlib import pyplot as plt
import numpy as np
import imageio
import glob
import os
from IPython.display import Image, display
from PIL import Image as ImagePIL


def gif_vol_cube_by_dates(data,
                          dates,
                          x_labels,
                          y_labels,
                          strikes
                          ):
    """
    Create a gif which display vol cube structure for different z0 / z1 values 

    
    Parameters:

    data: list of volatility cubes for different dates

    dates: the list of dates
    
    x_labels: list of labels of the x-axis on the grid graph (denotes swap tenors)

    y_labels: list of labels of the y-axis on the grid graph (denotes option tenors)

    uniq_strikes: list of strikes (for each strike, we construct a grid graph)

    strikes: list of all strikes in volatility data structure
    """

    SUBTITLE_SIZE = 10
    TICK_SIZE = 8

    folder_path = '../../reports/vol/gifs/'

    max_vol = data.max()
    min_vol = data.min()
    x_space = np.arange(len(x_labels))
    y_space = np.arange(len(y_labels))
    X, Y = np.meshgrid(x_space, y_space)
    strikes_len = len(strikes)

    for i, date in enumerate(dates):
        
        # Create plots
        fig = plt.figure(figsize=(16,3))
        date_string = f"{date.year}-{date.month:02}-{date.day:02}"
        fig.suptitle(f'vol cube data on ' + date_string + ' (in bp)')

        # Create subplots for each strike value
        for strk_idx, strk in enumerate(strikes):
            surface = data[i][:, :, strk_idx]
            ax = fig.add_subplot(1, strikes_len, strk_idx + 1, projection='3d')  # Create a 3D subplot    
            surf = ax.plot_surface(X, Y, surface, 
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
        file_path = folder_path + 'image_' + f"{i:04}" +'.png'
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
    gif_file_path = folder_path + 'plot_vol_cube_by_dates.gif'
    imageio.mimsave(gif_file_path, images, loop=0, fps=5)  # loop=0 means that gif will be played non-stop ; fps=100 is maximum

    # Delete all created png files
    for filename in sorted(glob.glob(folder_path + '*.png')):
        os.remove(filename)

    # Open gif
    display(Image(gif_file_path))   