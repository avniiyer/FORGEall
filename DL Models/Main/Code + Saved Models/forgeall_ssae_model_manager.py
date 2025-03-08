# -*- coding: utf-8 -*-

import tensorflow as tf
from itertools import product
from datetime import datetime
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

class SupervisedSparseAutoencoder:
    def __init__(self, hidden_dim, latent_dim, input_dim, target_sparsity=0.05):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.target_sparsity = target_sparsity
        self._build_model()

    def _build_model(self):
        # Input layer
        input_data = keras.layers.Input(shape=(self.input_dim,))

        # Encoder
        encoded = keras.layers.Dense(self.hidden_dim, activation='selu')(input_data)
        latent  = keras.layers.Dense(self.latent_dim, activation='sigmoid', name='latent_layer')(encoded)

        # Decoder
        decoded = keras.layers.Dense(self.hidden_dim, activation='selu')(latent)
        output_data = keras.layers.Dense(self.input_dim, activation='sigmoid')(decoded)

        # Prediction Layer (Supervised Component)
        prediction = keras.layers.Dense(1, activation='relu')(latent)

        # Autoencoder Model with Two Outputs
        self.autoencoder = keras.Model(inputs=input_data, outputs=[output_data, prediction])
        self.encoder = keras.Model(inputs=input_data, outputs=latent)

    def kl_divergence_loss(self, target_sparsity, actual_sparsity):
        return target_sparsity * tf.math.log(target_sparsity / (actual_sparsity + 1e-10)) + \
               (1 - target_sparsity) * tf.math.log((1 - target_sparsity) / (1 - actual_sparsity + 1e-10))

    def sparse_loss(self, y_true, y_pred):
        # Reconstruction loss (Mean Squared Error)
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

        # Calculate the average activation of the latent layer
        latent_output = self.encoder(y_true)
        actual_sparsity = tf.reduce_mean(latent_output, axis=0)

        # KL Loss
        kl_loss = tf.reduce_sum(self.kl_divergence_loss(self.target_sparsity, actual_sparsity))
        return reconstruction_loss + kl_loss

    def compile(self, optimizer='adam', loss_weights=None):
        self.autoencoder.compile(optimizer=optimizer,
                                 loss=[self.sparse_loss, 'mean_squared_error'],
                                 loss_weights=loss_weights)  # Adjust loss weights as needed

    def fit(self, train_data, train_labels, epochs=10, batch_size=256, validation_data=None, callbacks=None):
        return self.autoencoder.fit(train_data,
                                    [train_data, train_labels],
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    validation_data=validation_data,
                                    callbacks=callbacks)

    def evaluate_sparsity(self, x_data):
        latent_representations = self.encoder.predict(x_data)
        sparsity = np.mean(np.abs(latent_representations) < 1e-3)
        return sparsity * 100

    def predict_latent(self, x_data):
        return self.encoder.predict(x_data)

    def decoded_output(self, x_data):
        return self.autoencoder.predict(x_data)

    def fine_tune_regression(self, train_data, train_labels, epochs=5, learning_rate=1e-4):
        """
        Option to fine-tune the latent space for the regression task.
        - Freeze encoder weights and only update the regression output layer.
        """
        for layer in self.encoder.layers:
            layer.trainable = False  # Freeze encoder layers

        # Compile again with a smaller learning rate and only train the regression output
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                                 loss=[self.sparse_loss, 'mean_squared_error'],
                                 loss_weights=[0.4, 0.4])  # Equal weight for both losses

        # Fit only on regression
        self.autoencoder.fit(train_data,
                             [train_data, train_labels],  # Same inputs but focus is on regression
                             epochs=epochs,
                             shuffle=True)

def fa_train_ssae_model(train_data, test_data, train_labels, test_labels, loss_weight_divergence, loss_weight_sparse):
    # Reshape labels for Keras (they need to be 2D arrays)
    # Reshape labels for Keras (they need to be 2D arrays)
    train_labels_autoencoder = train_labels.values.reshape(-1, 1) # Use .values to get the NumPy array from the DataFrame
    test_labels_autoencoder = test_labels.values.reshape(-1, 1) # Use .values to get the NumPy array from the DataFrame

    # Define the strategy
    strategy = tf.distribute.MirroredStrategy()

    # Define the parameter grid
    param_grid = {
        'hidden_dim': [16, 18],
        'latent_dim': [6],
        'target_sparsity': [0.08],
        'batch_size': [4]
    }

    # Generate all combinations
    param_combinations = list(product(param_grid['hidden_dim'],
                                      param_grid['latent_dim'],
                                      param_grid['target_sparsity'],
                                      param_grid['batch_size']))

    # List to store results
    results               = []

    # Loop over all combinations
    for hidden_dim, latent_dim, target_sparsity, batch_size in param_combinations:
        print(f'Training with hidden_dim={hidden_dim}, latent_dim={latent_dim}, '
              f'target_sparsity={target_sparsity}, batch_size={batch_size}')
        # Clear previous session
        tf.keras.backend.clear_session()

        with strategy.scope():
            # Build the model
            sparse_autoencoder = SupervisedSparseAutoencoder(input_dim=25,  # Adjust input_dim as needed
                                                            hidden_dim=hidden_dim,
                                                            latent_dim=latent_dim,
                                                            target_sparsity=target_sparsity)
            sparse_autoencoder.compile(optimizer='adam', loss_weights=[loss_weight_sparse, loss_weight_divergence])

            # Fit the model
            history = sparse_autoencoder.fit(train_data,
                                            train_labels_autoencoder,
                                            epochs=32,
                                            batch_size=batch_size,
                                            validation_data=(test_data, [test_data, test_labels_autoencoder]))

        # Get validation loss from history
        val_loss = history.history['val_loss'][-1]
        #val_reconstruction_loss = history.history['val_reconstruction_output_loss'][-1]
        #val_prediction_loss = history.history['val_prediction_output_loss'][-1]

        # Record the results
        results.append({
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'target_sparsity': target_sparsity,
            'batch_size': batch_size,
            'val_loss': val_loss
            #'val_reconstruction_loss': val_reconstruction_loss,
            #'val_prediction_loss': val_prediction_loss
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Find the best combination
    best_result = results_df.loc[results_df['val_loss'].idxmin()]

    print("Best parameters:")
    print(f"hidden_dim: {best_result['hidden_dim']}")
    print(f"latent_dim: {best_result['latent_dim']}")
    print(f"target_sparsity: {best_result['target_sparsity']}")
    print(f"batch_size: {best_result['batch_size']}")
    print(f"Validation loss: {best_result['val_loss']}")

    return best_result

def fa_create_fit_and_save_model_to_file(model_filename_to_save, best_result, train_data, test_data, train_labels, test_labels, input_dim, hidden_dim, latent_dim, target_sparsity):
    train_labels_autoencoder = train_labels.values.reshape(-1, 1) # Use .values to get the NumPy array from the DataFrame
    test_labels_autoencoder  = test_labels.values.reshape(-1, 1) # Use .values to get the NumPy array from the DataFrame
    # Define the strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      sparse_autoencoder_to_disk = SupervisedSparseAutoencoder(input_dim=25, hidden_dim=int(best_result['hidden_dim']), latent_dim=int(best_result['latent_dim']), target_sparsity=best_result['target_sparsity'])
      sparse_autoencoder_to_disk.compile(optimizer='adam')
      history = sparse_autoencoder_to_disk.fit( train_data,
                                                train_labels_autoencoder,
                                                epochs=32,
                                                batch_size=4,
                                                validation_data=(test_data, [test_data, test_labels_autoencoder]))
                                                
    sparse_autoencoder_to_disk.autoencoder.save(model_filename_to_save)                                          

def fa_load_model_from_file(model_file):
    # Create your distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
      # Load the model for inference only (no custom objects needed)
      __loaded_model = keras.models.load_model(model_file, compile=False)
      # Verify the model summary after loading
      __loaded_model.summary()

      # Extract the encoder by specifying the latent layer's name.
      # Ensure that when building the model you named the latent layer with name='latent_layer'
      latent_layer  = __loaded_model.get_layer('latent_layer').output
      __models_latent_layer_encoding = keras.Model(inputs=__loaded_model.input, outputs=latent_layer)
      
      return __loaded_model, __models_latent_layer_encoding

def fa_get_latent_prediction_vectors_from_model(model_latent_layer_encoding, data):
      # Obtain latent representations from the encoder
      __latent_data = model_latent_layer_encoding.predict(data)
      # Scale these latent features
      scaler = RobustScaler()
      __latent_data_scaled = scaler.fit_transform(__latent_data)
      return __latent_data, __latent_data_scaled

      


