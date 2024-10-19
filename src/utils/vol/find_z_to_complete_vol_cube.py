from scipy.optimize import fmin
import numpy as np

from references.global_parameters import MISSED_VALUE


def find_z_to_complete_vol_cube(vae,
                                data,
                                random_attempt_num=30,
                                random_seed=0,
                                print_status=True
                                ):
    """
    Find the best z value for which the error between a reconstructed vol cube data and the real vol cube data is minimal

    
    Parameters:

    vae: The variational autoencoder model (keras.Model)
    
    data: Non-normalized vol cube data to be displayed on the grid graph

    random_attempt_num: The number of attempts when we start a search from a random z value
    
    random_seed: Random seed

    print_status: If True, the details of all iteration attempts are printed


    Methods:

    mse_func: Calculates mean squared error between generated vol cube from a certain z value and the real volatility cube

    
    Return:

    z_optimal: the optimal z value for which the error between a reconstructed vol cube data and the real vol cube data is minimal
    
    """

    def mse_func(z, vae, data):
        predictions = vae.decoder.predict(np.array([z]), verbose=0)[0]  # vae.decoder.predict has shape=(1,6,5,7)
        predictions[data == MISSED_VALUE] = MISSED_VALUE
        return ((predictions- data)**2).sum()

    np.random.seed(random_seed)

    latent_space_dim = vae.latent_space_dim  # data has (6,5,7) dimension
    mse = float('inf')

    for i in range(random_attempt_num):
        # Create n-dimensional gaussian variable z_initial
        mean = np.zeros(latent_space_dim)
        cov = np.eye(latent_space_dim) * 9  # variance=9
        z_initial = np.random.multivariate_normal(mean, cov, 1)[0]
        if print_status:
            print(f'iteration #{i}')
            print('z_initial', z_initial)
        result = fmin(mse_func, z_initial, 
                        args=(vae, data),
                        disp=False, # not to display status of convergence
                        full_output=True)  # to return minimum function value
        if result[1] < mse:
            mse = result[1]
            z_optimal = result[0]
        if print_status:
            print('z_optimal', z_optimal)
            print('current mse', result[1])
            print('best mse', mse)
            print('----------------------')

    return z_optimal