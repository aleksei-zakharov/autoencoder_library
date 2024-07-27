import numpy as np
import matplotlib.pyplot as plt


def error_histograms(model,
                     data,
                     normalizer):


    data_norm = normalizer.normalize(data)
    predictions = normalizer.denormalize(model.predict(x=data_norm, verbose=0))

    avg_errors = np.zeros(data.shape[0])
    max_errors = np.zeros(data.shape[0])

    for i, val in enumerate(data):
        diff = val - predictions[i]
        avg_errors[i] = (diff**2).mean()**0.5  # error is deviation from zero (not from mean)
        max_errors[i] = (diff**2).max()**0.5  # error is deviation from zero (not from mean)


    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].hist(avg_errors, 40)
    ax[0].set_title('distribution of the Average Error between real data and predictions')
    ax[0].set_xlabel('bp')

    ax[1].hist(max_errors, 40)
    ax[1].set_title('distribution of the Max Error between real data and predictions')
    ax[1].set_xlabel('bp')
    plt.show()


    # ADD SAVING TO THE FOLDER