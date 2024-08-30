import keras
import tensorflow as tf
import numpy as np

from keras.layers import Input, Flatten, Reshape, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import ops

from src.models.sampling import Sampling


class VaeVanilla(Model):
    def __init__(self,
                 input_shape,
                 hidden_layers_nodes,  # list,
                 latent_space_dim,
                 loss_type='bce',  # bce or mse
                 beta=1):  # recommendation from the article is beta = 4 for bce
        super().__init__()

        self.input_shape = input_shape
        self.hidden_layers_nodes = hidden_layers_nodes
        self.latent_space_dim = latent_space_dim
        self.loss_type = loss_type
        self.beta = beta
        
        self.encoder = None
        self.decoder = None
        self.model_type = 'vae'  # to plot functions universal for vae and ae
        self._hidden_layers_num = len(hidden_layers_nodes)
        self._build()


    def call(self, inputs):
        __, __, z = self.encoder(inputs)  # returns z_mean, z_log_var, z
        reconstruction = self.decoder(z)  # the output of our autoencoder
        return reconstruction


    @property
    def metrics(self):  # metrics that we track during the training
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):  # data is 1 object - x_train digits
        with tf.GradientTape() as tape:
            # Reconstruction loss
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            if self.loss_type == 'mse':
                reconstruction_loss = ops.mean(ops.square(Flatten()(data) - Flatten()(reconstruction)), axis=1)
            elif self.loss_type == 'bce':
                reconstruction_loss = binary_crossentropy(Flatten()(data), Flatten()(reconstruction))
                reconstruction_loss *= np.prod(self.input_shape)
            # Kullback-leibler loss
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = self.beta * ops.sum(kl_loss, axis=1)
            # Total loss
            total_loss = ops.mean(reconstruction_loss + kl_loss)
        # Gradient calculation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
                "total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                }


    def test_step(self, data):  # data is 1 object - x_train digits
        # the same as train step but there is no applying gradients here
        # without this step validation_data=(x_test, x_test) provided the ValueError: No loss to compute. Provide a loss argument in compile().
        with tf.GradientTape():
            # Reconstruction loss
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            if self.loss_type == 'mse':
                reconstruction_loss = ops.mean(ops.square(Flatten()(data) - Flatten()(reconstruction)), axis=1)
            elif self.loss_type == 'bce':
                reconstruction_loss = binary_crossentropy(Flatten()(data), Flatten()(reconstruction))
                reconstruction_loss *= np.prod(self.input_shape)
            # Kullback-leibler loss
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = self.beta * ops.sum(kl_loss, axis=1)
            # Total loss
            total_loss = ops.mean(reconstruction_loss + kl_loss)
        # Update losses
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
                "total_loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                }
    

    def _build(self):
        keras.utils.set_random_seed(0)
        input_image = Input(shape=(self.input_shape))  
        # Create a graph of calculation for encoder using keras.layers.Dense layers
        encoded = input_image
        encoded = Flatten()(encoded)

        for nodes in self.hidden_layers_nodes:
            encoded = Dense(nodes, activation='relu')(encoded)  

        z_mean = Dense(self.latent_space_dim)(encoded)  # can be from -inf to +inf
        z_log_var = Dense(self.latent_space_dim)(encoded)  # can be from -inf to +inf
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(inputs=input_image, 
                             outputs=[z_mean, z_log_var, z])
        
        # Create a graph of calculation for decoder
        decoded_input = Input(shape=(self.latent_space_dim,))
        decoded = decoded_input
        
        for nodes in reversed(self.hidden_layers_nodes):  # reverse order of hidden layers
            decoded = Dense(nodes, activation='relu')(decoded)  # because inputs are from 0 to 1
    
        flatten_input_dim = np.prod(self.input_shape)
        decoded = Dense(flatten_input_dim, activation='sigmoid')(decoded)
        decoded = Reshape(self.input_shape)(decoded)
        self.decoder = Model(inputs=decoded_input, 
                             outputs=decoded)

        # Calculate losses as mean values over all samples
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    def get_config(self):   # to save a model to a file
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'hidden_layers_nodes': self.hidden_layers_nodes,
            'latent_space_dim': self.latent_space_dim,
            'loss_type': self.loss_type,
            'beta': self.beta
        })
        return config


    @classmethod  # to save a model to a file
    def from_config(cls, config):
        expected_args = ['input_shape', 'hidden_layers_nodes', 'latent_space_dim', 'loss_type', 'beta']
        filtered_config = {k: v for k, v in config.items() if k in expected_args}
        return cls(**filtered_config)


if __name__ == "__main__":
    pass