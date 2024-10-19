import matplotlib.pyplot as plt
import os
from numpy import log


def plot_history_of_losses(history,
                           use_log=True,
                           data_type=None,
                           save_name=None):
    """
    Plot history of log of losses for train and test dataset

    
    Parameters:

    history: history from keras.model fitting

    use_log: if True, we use logarithms of total and reconstruction losses.
    
    data_type: type of data to be plotted. Possible values: 'mnist' or 'vol'. The plot is saved in the folder with this name

    save_name: the name of the trained model that is used here to name the saved plot. If it is not None, the plot is saved in the folder 
    """

    if use_log:
        total_loss = log(history.history['total_loss'])
        val_total_loss = log(history.history['val_total_loss'])
        if 'reconstruction_loss' in history.history:
            reconstruction_loss = log(history.history['reconstruction_loss'])
            val_reconstruction_loss = log(history.history['val_reconstruction_loss'])
    else:
        total_loss = history.history['total_loss']
        val_total_loss = history.history['val_total_loss']
        if 'reconstruction_loss' in history.history:
            reconstruction_loss = history.history['reconstruction_loss']
            val_reconstruction_loss = history.history['val_reconstruction_loss']

    # Graph shows total, reconstruction and KL losses for train and test datasets
    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 2, 1)
    main_color = 'olive'
    add_color = 'darkseagreen' # 'yellowgreen'
    ax1.set_xlabel('epochs')
    ax1.plot(total_loss, color=main_color, label='total_loss')
    ax1.plot(val_total_loss, color=add_color, label='val_total_loss')
    ax1.tick_params(axis='y', labelcolor=main_color)
    ax1.legend(loc='upper right')
    if use_log:
        ax1.set_title('logarithm of total loss')
        ax1.set_ylabel('log(total loss)', color=main_color)
    else:
        ax1.set_title('total loss')
        ax1.set_ylabel('total loss', color=main_color)

    # create graph with reconstruction and KL losses only for VAE; no graphs for AE
    if 'reconstruction_loss' in history.history:  
        ax21 = plt.subplot(1, 2, 2)
        main_color = 'tab:red'
        add_color = 'salmon'
        ax21.set_xlabel('epochs')
        ax21.plot(reconstruction_loss, color=main_color, label='reconstruction_loss')
        ax21.plot(val_reconstruction_loss, color=add_color, label='val_reconstruction_loss')
        ax21.tick_params(axis='y', labelcolor=main_color)
        ax21.legend(loc='upper left')
        if use_log:
            ax21.set_title('logarithm of reconstruction loss and KL loss')
            ax21.set_ylabel('log(recon loss)', color=main_color)
        else:
            ax21.set_title('reconstruction loss and KL loss')
            ax21.set_ylabel('recon loss', color=main_color)

        ax22 = ax21.twinx()  # instantiate a second Axes that shares the same x-axis
        main_color = 'tab:blue'
        add_color = 'lightblue'
        ax22.set_ylabel('KL loss', color=main_color)  # we already handled the x-label with ax1
        ax22.plot(history.history['kl_loss'], color=main_color, label='kl_loss')
        ax22.plot(history.history['val_kl_loss'], color=add_color, label='val_kl_loss')
        ax22.tick_params(axis='y', labelcolor=main_color)
        ax22.legend(loc='upper right')
    plt.tight_layout()

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