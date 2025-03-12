# -*- coding: utf-8 -*-
"""forgeall_data_prep.py
Prepare the HEA data and split it into training and testing sets: 
        train_data, test_data, train_labels, test_labels
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Step 1: Load and clean the dataset
def __load_and_clean_data(file_path, target_column, drop_columns_list):
    """
    Loads the dataset, drops irrelevant columns, calculates correlation matrix, and drops columns with high correlation.

    Args:
    - file_path (str): Path to the CSV file.
    - columns_to_drop (list): List of column names to drop initially.
    - target_column (str): Column to calculate correlation against.
    - drop_columns_list (list): Columns to drop based on analysis.

    Returns:
    - df (DataFrame): Cleaned dataframe.
    - X_train (DataFrame): Feature data without target column.
    - y_train (Series): Target column data.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop irrelevant columns
    #AI - df = df.drop(columns=columns_to_drop, axis=1)

    # Display correlation matrix (optional step for analysis)
    corr_matrix = df.corr()
    print(corr_matrix[target_column].sort_values(ascending=False))

    # Drop columns based on analysis
    #AI - df = df.drop(columns=drop_columns_list, axis=1)

    # Split features and target
    X_train = df.drop(target_column, axis=1)
    y_train = df[target_column]

    return df, X_train, y_train

# Step 2: Convert atomic percentage to weight percentage
def __convert_atomic_to_weight(X_train):
    """
    Molar mass is the mass of one mole of a given substance, which can be an
    element or a compound. It is typically expressed in grams per mole (g/mol).
    The molar mass of a substance is equivalent to the sum of the atomic masses of all atoms
    in a molecule of that substance, as represented in atomic mass units (amu), but
    converted to grams for use in macroscopic quantities.

    For elements:
    The molar mass of an element is numerically equal to its atomic mass
    (in atomic mass units), but in grams per mole. For example, the molar mass of carbon (C)
    is approximately 12.01 g/mol, because one mole of carbon atoms weighs about 12.01 grams.

    This function converts atomic percentages to weight percentages for relevant elements.
    currently, mol ratios written as a percentage of the whole. Dividing by 100 and multiplying
    by the molar mass gives the molar weight with respect to other elements

    Args:
    - X_train (DataFrame): Feature dataframe with atomic percentages.

    Returns:
    - X_train (DataFrame): Updated feature dataframe with weight percentages.
    """
    # Define atomic masses for each element
    atomic_masses = {
        'C(at%)': 12.01, 'Co(at%)': 58.93, 'Al(at%)': 26.98, 'V(at%)': 50.94,
        'Cr(at%)': 51.99, 'Mn(at%)': 54.94, 'Fe(at%)': 55.85, 'Ni(at%)': 58.69,
        'Cu(at%)': 63.55, 'Mo(at%)': 95.96
    }

    # Convert atomic percentages to weight percentages
    for element, mass in atomic_masses.items():
        X_train[element] = (X_train[element] / 100) * mass

    # Rename columns to indicate weight percentages
    X_train = X_train.rename(columns={f'{key}': key.replace('(at%)', '(wt)') for key in atomic_masses.keys()})

    X_train = X_train.rename(columns={
        'R(%)' : 'R',
        'CR(%)': 'CR'
    })
    # Normalize percentages for R and CR
    X_train['R'] = X_train['R'] / 100
    X_train['CR'] = X_train['CR'] / 100

    return X_train

# Step 3: Normalize features using MinMaxScaler
def __normalize_features(X_train, y_train):
    """
    Normalizes the feature and target data using MinMaxScaler.

    Args:
    - X_train (DataFrame): Feature data.
    - y_train (Series): Target data (Yield Strength).

    Returns:
    - X_train_normalized (DataFrame): Normalized feature data.
    - y_train_normalized (DataFrame): Normalized target data.
    """
    scaler_minmax = MinMaxScaler()

    # Normalize features
    X_train_normalized = scaler_minmax.fit_transform(X_train)
    X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)

    # Normalize target
    y_train_normalized = scaler_minmax.fit_transform(y_train.values.reshape(-1, 1))
    y_train_normalized = pd.DataFrame(y_train_normalized, columns=['YS(Mpa)'])

    return X_train_normalized, y_train_normalized

# Step 4: Check for missing values (NaN) in the dataset
def __check_missing_values(X_train):
    """
    Checks for missing values (NaN) in the dataset and prints out the columns with NaNs.

    Args:
    - X_train (DataFrame): Feature data.

    Returns:
    - None
    """
    nan_counts = X_train.isna().sum()
    columns_with_nan = X_train.columns[X_train.isna().any()].tolist()

    print("NaN counts per column:")
    print(nan_counts)
    print("\nColumns with NaN values:")
    print(columns_with_nan)

# Step 5: Split the data into training and testing sets
def __split_data(X_train_normalized, y_train_normalized, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
    - X_train_normalized (DataFrame): Normalized feature data.
    - y_train_normalized (DataFrame): Normalized target data.
    - test_size (float): Proportion of data to include in the test set.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - train_data (DataFrame): Training feature data.
    - test_data (DataFrame): Testing feature data.
    - train_labels (DataFrame): Training target data.
    - test_labels (DataFrame): Testing target data.
    """
    return train_test_split(X_train_normalized, y_train_normalized, test_size=test_size, random_state=random_state)

"""#<< - - - - Prepare the HEA data and split it into training and testing sets: train_data, test_data, train_labels, test_labels - - - - - - >>"""

# Main function to execute the pipeline
def fa_prep_hea_data(file_path):
    # File path to the dataset
    #file_path = 'OpenCalphad_w_melting_point - A.csv'

    # Columns to drop during cleaning
    #columns_to_drop = ['Enthalpy_BCC', 'Enthalpy_HCP', 'G_RT_BCC', 'G_RT_HCP',
    #                   'dG_RT_(BCC - FCC)', 'dG_RT_(HCP - FCC)', 'dG_RT_(BCC - HCP)',
    #                   'dG_AT_(BCC - FCC)', 'dG_AT_(HCP - FCC)', 'dG_AT_(BCC - HCP)',
    #                   'H_RT_BCC', 'H_RT_HCP']

    # Target column for prediction
    target_column = 'YS(Mpa)'

    # Additional columns to drop based on correlation analysis
    drop_columns_list = ['phase_fraction_hcp']

    # Load and clean the data
    df, X_train, y_train = __load_and_clean_data(file_path, target_column, drop_columns_list)

    # Convert atomic percentages to weight percentages
    X_train = __convert_atomic_to_weight(X_train)

    # Normalize features and target
    X_train_normalized, y_train_normalized = __normalize_features(X_train, y_train)

    # Check for missing values
    __check_missing_values(X_train)

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = __split_data(X_train_normalized, y_train_normalized)

    # Print shapes of train and test sets
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    return train_data, test_data, train_labels, test_labels

#train_data, test_data, train_labels, test_labels = fa_prep_hea_data()
#print_current_timestamp_in_pst()