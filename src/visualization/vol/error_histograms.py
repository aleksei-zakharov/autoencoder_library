import numpy as np
import matplotlib.pyplot as plt
import os


def error_histograms(predictions,  # not normalized
                     data,  # not normalized
                     save_name=None):

    errors = data - predictions
    errors = errors.flatten()

    print('Mean error',   round(np.std(errors), 2))
    print('Max error', round(np.max(abs(errors)), 2))

    plt.figure(figsize=(6,5))

    plt.hist(abs(errors), 40)
    plt.title('Histogram of absolute errors over all dates and data points')
    plt.xlabel('bp')
    plt.ylabel('observations')

    # Save plot
    if save_name is not None:
        # Make a folder
        folder_path = '../../reports/vol'
        os.makedirs(folder_path, exist_ok=True)  # make a folder if doesn't exist
        # Save plot if the file doesn't exist
        file_path = os.path.join(folder_path, save_name + '_error_histograms.png')
        plt.savefig(file_path)
    # Display the plot
    plt.show()