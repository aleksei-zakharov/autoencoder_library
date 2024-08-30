import matplotlib.pyplot as plt
import os
import numpy as np


def plot_history_of_losses(history,
                           data_type='mnist',  # type = 'mnist' or 'vol'
                           save_name=None):
    
    # graph shows logarithm of total losses for train and test datasets
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
        # Make a folder
        folder_path = '../../reports/' + data_type
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, save_name + 'history.png')
        plt.savefig(file_path)
    
    # Display plot
    plt.show()