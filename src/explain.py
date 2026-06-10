import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def get_shap_explainer_and_values(model, X_sample):
    """
    Dynamically creates a SHAP TreeExplainer and computes SHAP values for X_sample.
    
    HOW SHAP EXPLANATIONS ARE GENERATED:
    SHAP (SHapley Additive exPlanations) is based on Shapley values from cooperative game theory. 
    It treats each feature as a "player" in a game where the "payout" is the model's prediction. 
    For a given transaction, SHAP computes the marginal contribution of each feature to the 
    prediction across all possible feature subsets (coalitions).
    
    For tree-based models like XGBoost, we use TreeSHAP (TreeExplainer), which optimizes the 
    calculation by traversing the decision trees in polynomial time rather than exponential time.
    The output SHAP values represent the additive force (in log-odds space) that pushes the 
    prediction away from the base value (average model prediction on the background dataset) 
    towards the final prediction score.
    """
    print("Dynamically instantiating SHAP TreeExplainer on XGBoost model...")
    # Instantiate TreeExplainer directly on the trained model
    explainer = shap.TreeExplainer(model)
    
    print("Calculating SHAP values...")
    # Calculate SHAP values for the sample set
    shap_values = explainer(X_sample)
    
    # Handle binary classification multi-class output (if shape is (n_samples, n_features, 2))
    if len(shap_values.shape) == 3:
        print("Slicing SHAP explanation values for Class 1 (Fraud)...")
        shap_values = shap_values[:, :, 1]
        
    return explainer, shap_values

def plot_shap_summary(shap_values, X_sample, save_path=None):
    """
    Generates and saves a SHAP Summary Plot (beeswarm plot).
    
    WHAT THE SHAP SUMMARY PLOT REPRESENTS:
    The summary plot displays the global feature importance combined with local effects. 
    - Features are ranked vertically by their average absolute SHAP values (overall impact).
    - For each feature, each dot represents a single transaction.
    - The dot's horizontal position shows its SHAP value: positive values (right) increase 
      the probability of fraud, while negative values (left) decrease it.
      - The color represents the actual value of the feature (red for high, blue for low).
    For example, if V14 is red and on the left, it means high values of V14 reduce fraud risk.
    """
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    
    # Generate summary beeswarm plot
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Global Feature Impact (Beeswarm Plot)", pad=20, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP Summary Plot saved to {save_path}")

def plot_local_waterfall(shap_values, index, sample_name="transaction", save_path=None):
    """
    Generates and saves a SHAP Waterfall Plot for a specific sample index.
    
    WHAT THE SHAP WATERFALL PLOT REPRESENTS:
    The waterfall plot displays the local feature attributions for a single transaction.
    - It starts at the bottom at E[f(X)], which is the expected/base model output (log-odds).
    - Moving upwards, it shows how each feature's value pushes the prediction higher 
      (red bars, positive contribution) or lower (blue bars, negative contribution).
    - The top shows f(x), the final predicted log-odds score for this specific transaction.
    This provides an audit trail explaining exactly why a transaction was flagged or cleared.
    """
    print(f"Generating SHAP Waterfall Plot for {sample_name}...")
    plt.figure(figsize=(9, 6))
    
    # Generate waterfall plot
    shap.plots.waterfall(shap_values[index], max_display=10, show=False)
    plt.title(f"SHAP Local Explanation Waterfall ({sample_name.capitalize()})", pad=20, fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP Waterfall Plot saved to {save_path}")

def plot_local_force(explainer, shap_values, index, X_sample, sample_name="transaction", save_path=None):
    """
    Generates and saves a SHAP Force Plot for a specific sample index.
    
    WHAT THE SHAP FORCE PLOT REPRESENTS:
    Like the waterfall plot, the force plot shows the local feature attributions for a single sample.
    - Features that push the prediction higher (increase fraud probability) are shown in red.
    - Features that push the prediction lower (decrease fraud probability) are shown in blue.
    - The size of each arrow represents the magnitude of the feature's influence.
    - The meeting point between red and blue is the final predicted output value.
    """
    print(f"Generating SHAP Force Plot for {sample_name}...")
    
    # We extract the expected value and shap values for the specific sample
    # Force plot is typically generated as HTML, but we can save it as an image using matplotlib=True
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray) and len(expected_value.shape) > 0:
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
    s_vals = shap_values.values[index]
    feat_vals = X_sample.iloc[index]
    
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        expected_value, 
        s_vals, 
        feat_vals, 
        feature_names=X_sample.columns.tolist(),
        matplotlib=True, 
        show=False
    )
    plt.title(f"SHAP Local Explanation Force Plot ({sample_name.capitalize()})", pad=20, fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP Force Plot saved to {save_path}")

def run_shap_pipeline(model, X_test, y_test):
    """
    Runs the SHAP explanation pipeline on a balanced subset of the test set.
    """
    # Select all frauds in the test set and a subset of legitimate transactions
    # This prevents the calculations from being extremely slow, while keeping representation high
    frauds_idx = np.where(y_test == 1)[0]
    legits_idx = np.where(y_test == 0)[0]
    
    # Sample 400 legitimate transactions and combine with all fraud transactions
    np.random.seed(config.RANDOM_STATE)
    sampled_legits_idx = np.random.choice(legits_idx, size=400, replace=False)
    
    combined_idx = np.concatenate([frauds_idx, sampled_legits_idx])
    X_sample = X_test.iloc[combined_idx].copy()
    y_sample = y_test.iloc[combined_idx].copy()
    
    # Compute SHAP explainer and values
    explainer, shap_values = get_shap_explainer_and_values(model, X_sample)
    
    # Save Global summary plot
    plot_shap_summary(shap_values, X_sample, save_path=os.path.join(config.FIGURES_DIR, "shap_summary.png"))
    
    # Find a sample fraud transaction and a sample legitimate transaction to generate local explanations
    test_frauds = np.where(y_sample == 1)[0]
    test_legits = np.where(y_sample == 0)[0]
    
    if len(test_frauds) > 0:
        fraud_local_idx = test_frauds[0] # Pick the first fraud transaction in X_sample
        plot_local_waterfall(shap_values, fraud_local_idx, "fraudulent_transaction", 
                             save_path=os.path.join(config.FIGURES_DIR, "shap_waterfall.png"))
        plot_local_force(explainer, shap_values, fraud_local_idx, X_sample, "fraudulent_transaction", 
                         save_path=os.path.join(config.FIGURES_DIR, "shap_force.png"))
    else:
        print("Warning: No fraud transactions found in sample to generate local waterfall/force plots.")
        
    print("SHAP analysis pipeline completed successfully.")
