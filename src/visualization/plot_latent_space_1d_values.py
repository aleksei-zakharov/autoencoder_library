import matplotlib.pyplot as plt
from matplotlib import cm  # to change color scheme
import numpy as np


def plot_latent_space_1d_values(model, x, y=None):
    if model.model_type == 'vae':
        x_encoded, __, __ = model.encoder.predict(x, verbose=0)  # z_mean, z_var, z
    elif model.model_type == 'ae':
        x_encoded = model.encoder.predict(x, verbose=0)  # z

    bins = np.linspace(np.min(x_encoded), np.max(x_encoded), 400)
    
    if y is None:
        plt.hist(x_encoded, bins, alpha=0.5)
    else:
        for i in range(10):
            idxs = [key for key, val in enumerate(y) if val==i]
            plt.hist(x_encoded[idxs], bins, alpha=0.5, label=i)
        plt.legend()
    plt.show()