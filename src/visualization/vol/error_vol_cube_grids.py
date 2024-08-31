import numpy as np
import matplotlib.pyplot as plt
import os

from references.global_parameters import MIN_VOL_ON_GRAPHS, MAX_VOL_ON_GRAPHS


def error_vol_cube_grids(predictions,  
                        data,          
                        x_labels,
                        y_labels,
                        uniq_strikes, 
                        strikes, 
                        error_type='mse',
                        save_name=None):
    """
    Plot the error between the reconstructed volatility cubes and real vol cubes on several grid graphs (1 grid graph for 1 strike)

    
    Parameters:

    predictions: non-normalized reconstructed vol cube data
    
    data: non-normalized vol cube true data
    
    x_labels: list of labels of the x-axis on the grid graph (denotes swap tenors)

    y_labels: list of labels of the y-axis on the grid graph (denotes option tenors)
    
    uniq_strikes: list of strikes (for each strike, we construct a grid graph)

    strikes: list of all strikes in volatility data structure

    error_type: type of the error (possible values: 'mse', 'mean', 'abs_max')

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    TICK_SIZE = 10

    # Create grids for all strikes
    fig = plt.figure(figsize=(22,5.5))
    fig.suptitle(f'{error_type} error in bp (difference between predictions and real vol surfaces for different strikes)')

    for i_s, strk in enumerate(strikes):
        strk_idx = uniq_strikes.index(strk)

        # Create grid plot for certain strike     
        errors = np.zeros((len(y_labels), len(x_labels)))
        for i_y, y in enumerate(y_labels):      # option tenors
            for i_x, x in enumerate(x_labels):  # swap tenors
                # Calculate error for certain opt tenor and swap tenor
                diff = predictions[:, i_y, i_x, strk_idx] - data[:, i_y, i_x, strk_idx]
                if error_type == 'mse':
                    errors[i_y, i_x] = (diff**2).mean()**0.5
                elif error_type == 'abs_max':
                    errors[i_y, i_x] = abs(diff).max()
                elif error_type == 'mean':
                    errors[i_y, i_x] = diff.mean()
        
        ax = fig.add_subplot(1, len(strikes), i_s + 1)  # Create a 3D subplot    
        ax.set_title(strk)
        ax.matshow(errors, 
                   cmap=plt.get_cmap('Spectral_r'), 
                   vmin=MIN_VOL_ON_GRAPHS, 
                   vmax=MAX_VOL_ON_GRAPHS)
        ax.set_xticks(ticks=range(len(x_labels)), labels=x_labels, size=TICK_SIZE)
        ax.set_yticks(ticks=range(len(y_labels)), labels=y_labels, size=TICK_SIZE)
        if i_s == 0:
            ax.set_xlabel('swap tenors')
            ax.set_ylabel('option tenors')

        for (x, y), value in np.ndenumerate(errors):
            ax.text(y, x, f"{value:.0f}", va="center", ha="center", color='black')

    # Save plot
    if save_name is not None:
        # Make a folder if it doesn't exist
        folder_path = '../../reports/vol'
        os.makedirs(folder_path, exist_ok=True)
        # Save plot 
        file_path = os.path.join(folder_path, save_name + 'errors_grid_' + error_type + '.png')
        plt.savefig(file_path)

    # Display the plot
    plt.show()