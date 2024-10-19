import numpy as np
import matplotlib.pyplot as plt
import os

from references.global_parameters import MIN_VOL_ON_GRAPHS, MAX_VOL_ON_GRAPHS
from references.global_parameters import MISSED_VALUE


def grids_vol_cube(data,
                   x_labels,
                   y_labels,
                   strikes,
                   show_title=True,
                   save_name=None):
    """
    Plot volatility cube data on several grid graphs (1 grid graph for 1 strike)

    
    Parameters:

    data: non-normalized vol cube data to be displayed on the grid graph
    
    x_labels: list of labels of the x-axis on the grid graph (denotes swap tenors)

    y_labels: list of labels of the y-axis on the grid graph (denotes option tenors)
    
    uniq_strikes: list of strikes (for each strike, we construct a grid graph)

    strikes: list of all strikes in volatility data structure

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    TICK_SIZE = 10
    
    nan_val_flag = MISSED_VALUE in data   # denotes whether there are missed values in data
    
    # Create grid subplots for all strikes
    fig = plt.figure(figsize=(12,3.3))
    if show_title:
        if nan_val_flag:
            fig.suptitle(f'Vol cube data for different tenors and strikes with missed ({MISSED_VALUE}) values (in bp)')
        else:
            fig.suptitle(f'Vol cube data for different tenors and strikes (in bp)')

    for strk_idx, strk in enumerate(strikes):

        # Create grid plot for certain strike     
        errors = np.zeros((len(y_labels), len(x_labels)))
        for i_y, y in enumerate(y_labels):      # option tenors
            for i_x, x in enumerate(x_labels):  # swap tenors
                # Calculate error for certain opt tenor and swap tenor
                errors[i_y, i_x] = data[i_y, i_x, strk_idx]
        
        ax = fig.add_subplot(1, len(strikes), strk_idx + 1)  # Create a 3D subplot    
        ax.set_title(strk)
        ax.matshow(errors,
                    cmap=plt.get_cmap('Spectral_r'),
                    vmin=MIN_VOL_ON_GRAPHS,
                    vmax=MAX_VOL_ON_GRAPHS)
        ax.set_xticks(ticks=range(len(x_labels)), labels=x_labels, size=TICK_SIZE)
        ax.set_yticks(ticks=range(len(y_labels)), labels=y_labels, size=TICK_SIZE)
        if strk_idx == 0:
            ax.set_xlabel('swap tenors')
            ax.set_ylabel('option tenors')

        for (x, y), value in np.ndenumerate(errors):
            ax.text(y, x, f"{value:.0f}", va="center", ha="center", color='black')
    plt.tight_layout()

    # Save plot
    if save_name is not None:
        # Make a folder if it doesn't exist
        folder_path = '../../reports/vol'
        os.makedirs(folder_path, exist_ok=True)
        # Save plot
        if nan_val_flag:
            file_path = os.path.join(folder_path, save_name + '_grid_with_missed_vals.png')
        else:
            file_path = os.path.join(folder_path, save_name + '_grid.png')
        plt.savefig(file_path)

    # Display the plot
    plt.show()