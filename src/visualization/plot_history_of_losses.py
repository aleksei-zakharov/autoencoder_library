import matplotlib.pyplot as plt


def plot_history_of_losses(history):
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
    plt.show()