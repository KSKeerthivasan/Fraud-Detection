import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Ensure folders exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_samples.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")

# ML Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

# XGBoost Randomized Search Space
# We choose a comprehensive set of hyperparameters to optimize recall/precision trade-off (F1) and ROC-AUC.
XGB_PARAM_DIST = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1], # Note: Since we are using SMOTE to balance the training set, scale_pos_weight is kept at 1 to prevent double-correction.
}
