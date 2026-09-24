import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib

import sys
# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def load_data(path=config.RAW_DATA_PATH):
    """Loads the credit card transaction dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download it from Kaggle.")
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def check_missing_values(df):
    """Sanity checks dataset for missing values and reports status."""
    missing = df.isnull().sum().sum()
    print(f"Total missing values: {missing}")
    if missing > 0:
        print("Warning: Missing values detected. Dropping incomplete rows.")
        df = df.dropna()
    return df

def scale_and_split(df):
    """
    Splits the data into train and test sets, then scales Time and Amount.
    We use RobustScaler because both Time and Amount features exhibit highly skewed
    distributions and contain significant outliers. RobustScaler scales features using
    the median and Interquartile Range (IQR), making it robust to outliers.
    """
    # Convert Time to Hour for better generalization
    df["Hour"] = (df["Time"] % 86400) / 3600
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]
    
    # Perform a stratified train-test split to preserve the ratio of fraud cases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    print(f"Initial split completed:")
    print(f" - Train set size: {X_train.shape[0]} (Frauds: {sum(y_train)})")
    print(f" - Test set size: {X_test.shape[0]} (Frauds: {sum(y_test)})")
    
    # Scale Time and Amount columns
    scaler = RobustScaler()
    
    # Copy to avoid SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    cols_to_scale = ["Hour", "Amount"]
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Save the scaler for inference deployment
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"RobustScaler fitted and saved to {config.SCALER_PATH}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

def apply_smote(X_train, y_train):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to the training set.
    
    WHY SMOTE IS USED:
    Fraud detection datasets are extremely imbalanced (typically <0.2% fraud). Standard 
    classifiers trained on such data will default to predicting the majority class (legitimate) 
    to maximize accuracy, completely failing to detect fraud. 
    
    SMOTE solves this by synthesizing new minority class samples along the line segments 
    joining k-nearest neighbors of the minority class. This expands the decision region 
    of the fraud class and helps the model generalize better to unseen fraud cases, rather 
    than over-fitting by simply duplicating existing fraud samples (as in random over-sampling).
    
    IMPORTANT: We ONLY apply SMOTE to the training data. Applying it to the test data or before
    splitting would cause severe data leakage, inflating model metrics unrealistically.
    """
    print("Applying SMOTE to handle class imbalance in the training set...")
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE completed:")
    print(f" - Resampled training set size: {X_train_res.shape[0]}")
    print(f" - Resampled Class balance: Legitimate={sum(y_train_res==0)}, Fraud={sum(y_train_res==1)}")
    
    return X_train_res, y_train_res

def save_test_dataset_explorer_samples(X_test_raw, y_test):
    """
    Saves a selection of raw, unscaled test transactions to a CSV file.
    This CSV will be loaded by the Streamlit dashboard as a Test Dataset Explorer.
    We include all fraud cases from the test set and a subset of legitimate transactions.
    """
    test_df = X_test_raw.copy()
    test_df["Class"] = y_test
    
    # Extract all frauds in test set
    frauds = test_df[test_df["Class"] == 1]
    # Extract a random sample of legitimate transactions
    legits = test_df[test_df["Class"] == 0].sample(n=2000, random_state=config.RANDOM_STATE)
    
    # Combine and shuffle
    explorer_df = pd.concat([frauds, legits]).sample(frac=1.0, random_state=config.RANDOM_STATE)
    explorer_df.to_csv(config.TEST_DATA_PATH, index=True)
    print(f"Saved {explorer_df.shape[0]} test samples (including {frauds.shape[0]} frauds) to {config.TEST_DATA_PATH} for dashboard explorer.")

def preprocess_pipeline():
    """Main orchestrator for the preprocessing steps."""
    df = load_data()
    df = check_missing_values(df)
    
    X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = scale_and_split(df)
    
    # Save test explorer samples
    save_test_dataset_explorer_samples(X_test_raw, y_test)
    
    # Apply SMOTE to training data for baselines
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
    
    # Return both SMOTEd data (for baselines) and un-SMOTEd scaled data (for Pipeline CV)
    return X_train_res, y_train_res, X_train_scaled, X_test_scaled, y_train, y_test
