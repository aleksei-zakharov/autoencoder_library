# Autoencoder library
This library trains variational autoencoders (VAEs) on volatility cubes to compress, analyze, complete, and generate volatility cubes.

Also, it trains PCA models, autoencoders, and variational autoencoders on the MNIST dataset.


## The structure of the library

The library contains 5 folders:

* ***models***: the trained models and training history are saved here. 
  * 'mnist' folder is devoted to the models that work with the MNIST dataset
  * 'vol' folder consists of models trained on volatility cube dataset
  
* ***notebook***: the ipynb files with the main results can be found in this folder.
  * 'mnist extras' folder is devoted to the notebooks that work with the MNIST dataset and are *not important*
  * 'mnist main' folder is devoted to the *crucial* notebooks that work with the MNIST dataset
  * 'vol cube extras' folder consists of notebooks that work with the volatility cube dataset and are *not important*
  * 'vol cube main' folder consists of *crucial* notebooks where we train models on volatility cube dataset

* ***references***: contains the file with global parameters

* ***reports***: plot images and GIFs can be saved in this folder
  * 'mnist' folder is devoted to the images that work with the MNIST dataset
  * 'vol' folder consists of images and GIFs that deal with the volatility cube dataset

* ***src***: it consists of 4 folders which are dedicated to data preprocessing, model classes, utility functions, and visualization methods:
  * 'data' folder comprise normalizer function and other functions related to data preprocessing and data loading
  * 'model' folder contains classes of autoencoder and variational autoencoder models
  * 'utils' folder has functions that split dataset to train and test, load and save trained models
  * 'visualization' folder is dedicated to the functions related to data visualization



## Main results

The files with main results can be found in the notebooks folder.

* ***mnist main*** folder contains the following notebooks:
  * 'ae_vanilla_2d' displays the results of training the autoencoder model
  * 'pca' displays the results of training PCA model
  * 'vae_vanilla_2d' displays the results of training the variational autoencoder model

* ***vol cube main*** folder contains the following notebooks:
  * 'vae_van_compress_and_analyze' shows how the trained VAE compress and analyze volatility cube data
  * 'vae_van_complete_atm_vols' displays how the trained VAE can complete volatility cube data when there are missed ATM values
  * 'vae_van_complete_all_except_atm_vols' displays how the trained VAE can complete volatility cube data when all data are missed except ATM vols
  * 'vae_van_generate_and_check_alignment_with_test_datase' checks how good generated vol cubes are align with history
  * 'vae_van_generate_check_robustness' check VAE robustness to changing latent space variables values
  * 'vae_van_generate_gif_from_z_values' displays GIF with volatility cubes for different latent space variables values
  * 'vae_van_generate_training_on_small_vols' checks how good generated vol cubes are align with history when we train only on the periods on small vols
