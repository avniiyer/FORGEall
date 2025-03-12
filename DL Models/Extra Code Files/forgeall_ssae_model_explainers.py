# -*- coding: utf-8 -*-
"""forgeall_ssae_model_explainers.py

"""

import shap
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

def fa_explain_ssae_model_using_feature_correlations(best_knn, encoder_model, train_data, test_data, latent_train_scaled, latent_test_scaled, test_labels):
    # First Inspect Encoder Weights: Analyze the weights of the encoder to identify the weights of the LATENT_DIM features.  
    encoder_weights = encoder_model.get_weights()[0]  # Assuming the first weight matrix
    num_latent_features  = latent_train_scaled.shape[1]
    latent_feature_names = [f'Latent Feature {i+1}' for i in range(num_latent_features)]

    # Step 2: Compute Permutation Importances
    result = permutation_importance(estimator=best_knn,
                                    X=latent_test_scaled,
                                    y=test_labels,
                                    n_repeats=60,
                                    random_state=42,
                                    scoring='r2'
                                   )
    importances = result.importances_mean
    std = result.importances_std

    # Create DataFrame
    feature_importances = pd.DataFrame({
                                      'Feature': latent_feature_names,
                                      'Importance': importances,
                                      'Std': std
                                  })
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

    # Step 3: Interpret and Visualize
    print(f"Total latent features = {num_latent_features}")
    # Print feature importances
    print("Permutation Feature Importances:")
    print(feature_importances)

    # Then VISUALIZE the latent_dims and their weights
    # Plot importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], xerr=feature_importances['Std'])
    plt.xlabel('Permutation Importance (Decrease in R²)')
    plt.title('Feature Importances for KNN Regressor on the Latent Dims of the Original data')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    #FINALLY do the correlations
    encoder_weights.shape # Print the encoder weights shape just to get a feel for the geometry

    important_latent_indices = feature_importances.index[:num_latent_features]
    # Extract weights corresponding to important latent features
    important_weights = encoder_weights[:, important_latent_indices] # Extracts from the full encoder weight matrix only those columns corresponding
                                                                    # to the important latent features. This gives you a submatrix of shape [Noriginal, Nselected_latent]
    # Create a DataFrame
    # original_feature_names = [...]  # List of original feature names - This was incomplete
    original_feature_names = train_data.columns # Get the columns from the original dataframe
    weights_df = pd.DataFrame(important_weights, index=original_feature_names, columns=[latent_feature_names[i] for i in important_latent_indices])

    print("Weights from Original Features to Important Latent Features:")
    print(weights_df)

def fa_explain_ssae_model_using_shap(best_knn, encoder_model, train_data, test_data):
    # Scale these latent features
    scaler = RobustScaler()

    # Assume train_data and test_data are available.
    # If they are pandas DataFrames, we can extract column names.
    if hasattr(train_data, 'columns'):
        original_feature_names = train_data.columns.tolist()
    else:
        # Otherwise, define feature names manually.
        original_feature_names = [f"feature_{i}" for i in range(train_data.shape[1])]

    # Define a pipeline function that takes original inputs, passes them through
    # the encoder, applies the same scaling, and then uses the best KNN regressor.
    def model_pipeline(X):
        # Get latent representation using the pre-trained encoder
        latent = encoder_model.predict(X)
        # Scale the latent features with the fitted scaler (e.g., RobustScaler)
        latent_scaled = scaler.fit_transform(latent)
        # Predict using the tuned KNN regressor
        return best_knn.predict(latent_scaled)

    # For efficiency, select a small random sample from the training data as background.
    # KernelExplainer uses this sample to approximate the model's behavior.
    n_background = min(100, train_data.shape[0])
    background_idx = np.random.choice(train_data.shape[0], n_background, replace=False)
    background = train_data.iloc[background_idx]

    # Select some test samples for which to explain predictions.
    X_to_explain = test_data[:10]

    # Create the SHAP KernelExplainer with the model pipeline and background dataset.
    explainer = shap.KernelExplainer(model_pipeline, background)

    # Compute SHAP values for the selected test samples.
    # (KernelExplainer can be computationally expensive; adjust the sample size as needed.)
    shap_values = explainer.shap_values(X_to_explain)

    # Select the SHAP values for the prediction output (usually the second output of the autoencoder)
    shap_values = shap_values[:, :, 0]  # Assuming prediction output is at index 1

    # Visualize the overall feature importance using a summary plot.
    shap.summary_plot(shap_values, X_to_explain, feature_names=original_feature_names)

    # Optionally, you can also create a force plot for a single prediction (e.g., the first test sample).
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], X_to_explain.iloc[0],
                    feature_names=original_feature_names)