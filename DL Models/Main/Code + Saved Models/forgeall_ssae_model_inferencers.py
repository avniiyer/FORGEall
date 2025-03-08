# -*- coding: utf-8 -*-
"""forgeall_ssae_model_inferencers.py

"""

from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def fa_save_scikit_regressed_model_to_file(scikit_model, model_filename):
    joblib.dump(scikit_model, model_filename)

def fa_load_scikit_regressed_model_from_file(model_filename):
    __loaded_best_knn = joblib.load(model_filename)
    return __loaded_best_knn

def fa_perform_k_nearest_neighbor_regression(latent_train_scaled, latent_test_scaled, train_labels, test_labels):
    # Define parameter grid
    param_grid = {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors': range(1, 51, 2),  # Test odd values from 1 to 49
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2, 3]  # For Minkowski metric with different powers
    }

    # Grid search on latent data
    grid_search_knn = GridSearchCV(estimator=KNeighborsRegressor(),
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring='r2',
                                   n_jobs=-1
                                  )
    # Define the strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      grid_search_knn.fit(latent_train_scaled, train_labels)

    __best_knn = grid_search_knn.best_estimator_
    print("Best KNN Hyperparameters:", grid_search_knn.best_params_)

    # Evaluate best model on latent data
    __knn_predictions_test = __best_knn.predict(latent_test_scaled)

    print("\nBest KNN Regression Performance on Latent Test Data:")
    print("MSE:", mean_squared_error(test_labels, __knn_predictions_test))
    print("RMSE:", np.sqrt(mean_squared_error(test_labels, __knn_predictions_test)))
    print("MAE:", np.mean(np.abs(test_labels - __knn_predictions_test)))
    print("R²:", r2_score(test_labels, __knn_predictions_test))

    return __best_knn, __knn_predictions_test

def fa_perform_xgboost_regression(latent_train_scaled, latent_test_scaled, train_labels, test_labels):
    # Initialize model
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    # Fit model to training data
    xgb_model.fit(latent_train_scaled, train_labels)
    # Make predictions
    xgb_predictions = xgb_model.predict(latent_test_scaled)
    print("\nBest XGBoost Regression Performance on Latent Test Data:")
    print("MSE:", mean_squared_error(test_labels, xgb_predictions))
    print("RMSE:", np.sqrt(mean_squared_error(test_labels, xgb_predictions)))
    print("MAE:", np.mean(np.abs(test_labels.values - xgb_predictions)))
    print("R²:", r2_score(test_labels, xgb_predictions))

def fa_perform_svm_regression(latent_train_scaled, latent_test_scaled, train_labels, test_labels):
    # Initialize SVR with chosen hyperparameters (example values)
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')

    # Train the model
    svr_model.fit(latent_train_scaled, train_labels)

    # Evaluate the model on latent test data
    svr_predictions_test = svr_model.predict(latent_test_scaled)
    mse  = mean_squared_error(test_labels, svr_predictions_test)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(test_labels.values - svr_predictions_test))
    r2   = r2_score(test_labels, svr_predictions_test)

    print("\nSVR Regression Performance on Latent Test Data:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R²:", r2)

def fa_perform_kerasnn_regression(latent_train_scaled, latent_test_scaled, train_labels, test_labels):
    # Determine input dimension from the latent representations
    input_dim = latent_train_scaled.shape[1]

    # --- Step 3: Build a simple neural network model ---
    # Create the model within the strategy scope to ensure proper device placement
    with tf.distribute.MirroredStrategy().scope():  # or tf.distribute.OneDeviceStrategy('/cpu:0')
        model = Sequential()
        # First hidden layer with 64 neurons
        model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
        # Second hidden layer with 32 neurons
        model.add(Dense(32, activation='relu'))
        # Output layer for regression
        model.add(Dense(1))
        # Compile the model using mean squared error as loss
        model.compile(loss='mean_squared_error', optimizer='adam')

    # --- Step 4: Train the model ---
    history = model.fit(latent_train_scaled,
                        train_labels,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1
                       )
                       
    # --- Step 5: Evaluate the model on the latent test data ---
    predictions = model.predict(latent_test_scaled)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_labels.values - predictions))
    r2 = r2_score(test_labels, predictions)

    print("\nNeural Network Regression Performance on Latent Test Data:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R²:", r2)  
