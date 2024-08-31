from scipy.optimize import fmin
import numpy as np

from references.global_parameters import MISSED_VALUE


def find_z_to_complete_vol_cube(vae,
                                data,
                                random_attempt_num=30, # number of attempts starting from different random z initial values
                                random_seed=0,
                                print_status=True
                                ):

    def mse_func(z, vae, data, mse_or_max):
        predictions = vae.decoder.predict(np.array([z]), verbose=0)[0]  # vae.decoder.predict has shape=(1,6,5,7)
        n0 = data.shape[0]
        n1 = data.shape[1]
        n2 = data.shape[2]
        mse_ = 0
        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    if data[i,j,k] != MISSED_VALUE:
                        if mse_or_max == 'mse':
                            mse_ += (predictions[i,j,k] - data[i,j,k])**2
                        elif mse_or_max == 'max':
                            mse_ = max(mse_, abs(predictions[i,j,k] - data[i,j,k]))
        return mse_


    np.random.seed(random_seed)

    latent_space_dim = vae.encoder.predict(x=np.expand_dims(data,0), verbose=0)[0].shape[1]  # data has (6,5,7) dimension
    mse = float('inf')

    for i in range(random_attempt_num):
        # Create n-dimensional gaussian variable z_initial
        mean = np.zeros(latent_space_dim)
        cov = np.eye(latent_space_dim) * 4  # variance=4
        z_initial = np.random.multivariate_normal(mean, cov, 1)[0]
        if print_status:
            print(f'iteration #{i}')
            print('z_initial', z_initial)
        result = fmin(mse_func, z_initial, 
                        args=(vae, data, 'mse'),
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