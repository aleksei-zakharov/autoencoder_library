import matplotlib.pyplot as plt
import os
import numpy as np


def plot_history_of_losses(history,
                           data_type=None,
                           save_name=None):
    """
    Plot history of log of losses for train and test dataset

    
    Parameters:

    history: history from keras.model fitting
    
    data_type: type of data to be plotted. Possible values: 'mnist' or 'vol'. The plot is saved in the folder with this name

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    # Graph shows logarithm of total losses for train and test datasets
    plt.plot(np.log(history.history['total_loss']))
    if 'val_total_loss' in history.history.keys():
        plt.plot(np.log(history.history['val_total_loss']))
        plt.legend(['train', 'test'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')
    plt.title('model loss')
    plt.ylabel('log(loss)')
    plt.xlabel('epoch')

    # Save plot
    if save_name is not None:
        if data_type is None:
            raise NameError('data_type parameter was not defined (possible values are \'mnist\' or \'vol\')')
        else:
            # Make a folder if it doesn't exist
            folder_path = '../../reports/' + data_type
            os.makedirs(folder_path, exist_ok=True) 
            # Save plot
            file_path = os.path.join(folder_path, save_name + 'history.png')
            plt.savefig(file_path)
    
    # Display plot
    plt.show()