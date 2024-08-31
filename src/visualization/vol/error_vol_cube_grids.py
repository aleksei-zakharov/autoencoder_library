import numpy as np
import matplotlib.pyplot as plt
import os

from references.global_parameters import MIN_VOL_ON_GRAPHS, MAX_VOL_ON_GRAPHS


def error_vol_cube_grids(predictions,  # not normalized
                        data,          # not normalized
                        x_labels,
                        y_labels,
                        uniq_strikes, 
                        strikes, 
                        error_type='mse',   # mse, mean or abs_max
                        save_name=None):

    TICK_SIZE = 10

    # Create grids for all strikes
    fig = plt.figure(figsize=(22,5.5))
    fig.suptitle(f'{error_type} error in bp (difference between predictions and real vol surfaces for different strikes)')

    for i_s, strk in enumerate(strikes):
        strk_idx = uniq_strikes.index(strk)

        # Create grid plot for certain strike     
        errors = np.zeros((len(y_labels), len(x_labels)))
        for i_y, y in enumerate(y_labels):      # opt tenors
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