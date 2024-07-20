import keras
import tensorflow as tf
import numpy as np

from keras.layers import Input, Flatten, Reshape, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import ops


class AeVanilla(keras.Model):
    def __init__(self,
                 input_shape,
                 hidden_layers_nodes,  # list,
                 latent_space_dim,
                 loss_type='bce'  # bce or mse
                 ):   
         
        super().__init__()

        self.input_shape = input_shape
        self.hidden_layers_nodes = hidden_layers_nodes
        self.latent_space_dim = latent_space_dim
        self.loss_type = loss_type
        
        self.encoder = None
        self.decoder = None
        self.model_type = 'ae'  # to plot functions universal for vae and ae
        self._hidden_layers_num = len(hidden_layers_nodes)
        self._build()

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))
    

    @property
    def metrics(self):  # metrics that we track during the training
        return [
            self.loss_tracker       
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Loss
            predictions = self.decoder(self.encoder(data))
            if self.loss_type == 'bce':
                total_loss = ops.mean(ops.square(Flatten()(data) - Flatten()(predictions)), axis=1)
            elif self.loss_type == 'mse':
                total_loss = binary_crossentropy(Flatten()(data), Flatten()(predictions))
                total_loss *= np.prod(self.input_shape)
            total_loss = ops.mean(total_loss)
        # Gradient calculation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update losses
        self.loss_tracker.update_state(total_loss)
        return {
                "total_loss": self.loss_tracker.result()
                }


    def test_step(self, data):
        # without this step validation_data=(x_test, x_test) provided the ValueError: No loss to compute. Provide a loss argument in compile().
        with tf.GradientTape():
            # Loss
            predictions = self.decoder(self.encoder(data))
            if self.loss_type == 'bce':
                total_loss = ops.mean(ops.square(Flatten()(data) - Flatten()(predictions)), axis=1)
            elif self.loss_type == 'mse':
                total_loss = binary_crossentropy(Flatten()(data), Flatten()(predictions))
                total_loss *= np.prod(self.input_shape)
            total_loss = ops.mean(total_loss)
        # Update losses
        self.loss_tracker.update_state(total_loss)
        return {
                "total_loss": self.loss_tracker.result()
                }


    def _build(self):
        keras.utils.set_random_seed(0)
        input_image = Input(shape=(self.input_shape))  
        # Create a graph of calculation for encoder using keras.layers.Dense layers
        encoded = input_image
        encoded = Flatten()(encoded)
        for i in range(self._hidden_layers_num):
            encoded = Dense(self.hidden_layers_nodes[i], activation='relu')(encoded)  
        encoded = Dense(self.latent_space_dim)(encoded)  # can be from -inf to +inf
        self.encoder = Model(inputs=input_image,
                             outputs=encoded)
        
        # Create a graph of calculation for decoder
        decoded = encoded
        for i in range(self._hidden_layers_num-1, -1, -1):  # reverse order of hidden layers
            decoded = Dense(self.hidden_layers_nodes[i], activation='relu')(decoded)  # because inputs are from 0 to 1
        flatten_input_dim = np.prod(self.input_shape)
        decoded = Dense(flatten_input_dim, activation='sigmoid')(decoded)
        decoded = Reshape(self.input_shape)(decoded)
        self.decoder = Model(inputs=encoded, 
                             outputs=decoded)

        # Calculate losses as mean values over all samples
        self.loss_tracker = keras.metrics.Mean(name="loss")
        

    def get_config(self):   # to save a model to a file
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'hidden_layers_nodes': self.hidden_layers_nodes,
            'latent_space_dim': self.latent_space_dim,
            'loss_type': self.loss_type
        })
        return config


    @classmethod  # to save a model to a file
    def from_config(cls, config):
        expected_args = ['input_shape', 'hidden_layers_nodes', 'latent_space_dim', 'loss_type']
        filtered_config = {k: v for k, v in config.items() if k in expected_args}
        return cls(**filtered_config)


if __name__ == "__main__":
    pass