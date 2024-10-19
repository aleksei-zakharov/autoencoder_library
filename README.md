# Autoencoder library
This library trains variational autoencoders (VAEs) on volatility cubes to compress, analyze, complete, and generate volatility cubes.

Also, it trains PCA models, autoencoders, and variational autoencoders on the MNIST dataset.


## The structure of the library

The library contains 3 folders:
  
* ***'notebook'*** folder: the ipynb files with the main results can be found in this folder.
  * ***'mnist extras'*** folder is devoted to the notebooks that work with the MNIST dataset and are *not important*
  * ***'mnist main'*** folder is devoted to the *crucial* notebooks that work with the MNIST dataset:
  * ***'vol cube extras'*** folder consists of notebooks that work with the volatility cube dataset and are *not important*
  * ***'vol cube main'*** folder consists of *crucial* notebooks where we train models on volatility cube dataset:
    
* ***'references'*** folder: contains the file with global parameters

* ***'src'*** folder: it consists of 4 folders which are dedicated to data preprocessing, model classes, utility functions, and visualization methods:
  * ***'data'*** folder comprise normalizer function and other functions related to data preprocessing and data loading
  * ***'model'*** folder contains classes of autoencoder and variational autoencoder models
  * ***'utils'** folder has functions that split dataset to train and test, load and save trained models
  * ***'visualization'*** folder is dedicated to the functions related to data visualization



## Main results

The files with main results can be found in the notebooks folder.

* ***'mnist main'*** folder contains the notebooks which shows results for digit picture reconstructions and latent space distributions for each out of these 3 methods:
  * ***'ae_vanilla_2d'*** displays the results of training the autoencoder model
  * ***'pca'*** displays the results of training PCA model
  * ***'vae_vanilla_2d'*** displays the results of training the variational autoencoder model

* ***'vol cube main'*** folder contains the results of applying variational autoencoders to volatility cube dataset:
     * ***'calc_all_models_to_select_hyperparams_randomsplit_leakyrelu_1000ep'*** shows mean, max reconstruction errors and Kullback-Leibler loss for models with different latent space dimensions, betas and hidden layers' structures (for random splitting)
     *  ***'calc_all_models_to_select_hyperparams_temporalsplit_leakyrelu_1000ep'*** shows mean, max reconstruction errors and Kullback-Leibler loss for models with different latent space dimensions, betas and hidden layers' structures (for temporal splitting)
     *  ***'check_best_model_random_split_100_50_25_12_1e-5_3000ep'*** shows results of compressing, analyzing, completing and generating for the best model for random splitting
     *  ***'check_best_model_random_split_100_50_25_12_1e-5_3000ep'*** shows results of compressing, analyzing, completing and generating for the best model for temporal splitting
     *  ***'check_consistency_of_gen_cubes_over_all_dates'*** shows consistency of generated volatility cubes over all dates
