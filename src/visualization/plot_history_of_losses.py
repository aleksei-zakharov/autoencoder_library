import matplotlib.pyplot as plt
import os


def plot_history_of_losses(history,
                           save_name=None):
    
    # graph shows losses for train and test datasets
    plt.plot(history.history['total_loss'])
    if 'val_total_loss' in history.history.keys():
        plt.plot(history.history['val_total_loss'])
        plt.legend(['train', 'test'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    # Save plot
    if save_name is not None:
        # Make a folder
        folder_path = '../../reports/mnist'
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, save_name + 'history.png')
        if os.path.exists(file_path):
            raise FileExistsError(f"The file '{file_path}' already exists.")
        else:
            plt.savefig(file_path)
    
    # Display plot
    plt.show()