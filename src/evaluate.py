import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix, roc_curve, 
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculates all key model performance metrics."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    # PR-AUC is also known as Average Precision (AP)
    pr_auc = average_precision_score(y_true, y_prob)
    brier_score = brier_score_loss(y_true, y_prob)
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier_score
    }

def analyze_thresholds(y_true, y_prob):
    """
    Evaluates model performance at thresholds 0.30, 0.50, and 0.70.
    Demonstrates the trade-off between recall (finding all frauds) 
    and precision (minimizing false alarms).
    
    NOTE: In a true production environment, the optimal threshold should be selected 
    using a hold-out validation set or out-of-fold cross-validation predictions, 
    not the test set. Here we analyze the test set sensitivity for reporting purposes.
    """
    thresholds = [0.30, 0.50, 0.70]
    results = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred_t, zero_division=0)
        r = recall_score(y_true, y_pred_t)
        f = f1_score(y_true, y_pred_t)
        
        results.append({
            "Threshold": f"{t:.2f}",
            "Precision": p,
            "Recall": r,
            "F1 Score": f
        })
        
    df_results = pd.DataFrame(results)
    print("\n--- Threshold Optimization Comparison ---")
    print(df_results.to_string(index=False))
    return df_results

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Generates a professional, annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    # Using blue scale heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legitimate", "Fraudulent"],
                yticklabels=["Legitimate", "Fraudulent"], ax=ax)
    
    ax.set_title("Confusion Matrix (Threshold = 0.50)", pad=15, fontweight='bold')
    ax.set_xlabel("Predicted Class", labelpad=10)
    ax.set_ylabel("Actual Class", labelpad=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plots the ROC Curve and reports the ROC-AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", pad=15, fontweight='bold')
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """
    Plots the Precision-Recall (PR) Curve and reports the PR-AUC score.
    
    WHY PR-AUC IS IMPORTANT:
    In highly imbalanced datasets, the ROC curve can present a misleadingly optimistic
    picture because it relies on the False Positive Rate (FPR = FP / (FP + TN)). Since the number 
    of True Negatives (TN) is massive, a large increase in False Positives (FP) results in a barely 
    noticeable change in FPR, masking the increase in false alarms.
    
    The PR curve is threshold-independent but focuses only on the positive (minority) class. It 
    evaluates Precision (TP / (TP + FP)) against Recall (TP / (TP + FN)), excluding TN entirely.
    PR-AUC (Average Precision) captures how many predicted frauds are actually frauds versus how many 
    actual frauds we caught, making it the most realistic indicator of a model's true performance.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#d62728', lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})")
    
    # Calculate baseline (no skill classifier has PR-AUC equal to the ratio of positive class)
    baseline = sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='#7f7f7f', lw=1.5, linestyle="--", label=f"Baseline (AUC = {baseline:.4f})")
    
    ax.set_title("Precision-Recall (PR) Curve", pad=15, fontweight='bold')
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_calibration_curve_plot(y_true, y_prob, save_path=None):
    """
    Generates a Probability Calibration Curve and calculates the Brier Score.
    
    WHY CALIBRATION IS IMPORTANT:
    For a fraud detection system, raw probability scores are used directly to route
    transactions (e.g., probability > 70% goes to block, 30-70% to human review). If a model
    predicts a fraud probability of 70%, then 70% of those transactions should actually be fraudulent.
    If the model is uncalibrated (e.g. over-confident or under-confident), the risk score is misleading.
    The Brier Score measures the mean squared difference between predicted probability and actual class (lower is better).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    brier_score = brier_score_loss(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#9467bd', label=f"XGBoost (Brier Score = {brier_score:.4f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='#7f7f7f', label="Perfect Calibration")
    
    ax.set_title("Probability Calibration Curve", pad=15, fontweight='bold')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Fraud)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_xgb_feature_importance(model, feature_names, save_path=None):
    """
    Generates and saves the XGBoost Feature Importance plot based on Gain.
    
    Gain represents the fractional contribution of each feature to the model 
    (i.e., the reduction in training loss achieved by splitting on this feature).
    """
    # Get feature importance (gain by default in scikit-learn API)
    importances = model.feature_importances_
    
    # Sort them
    indices = np.argsort(importances)[::-1]
    
    # Let's select the top 15 features
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    feat_names_top = np.array(feature_names)[top_indices]
    sns.barplot(x=importances[top_indices], y=feat_names_top, hue=feat_names_top, palette="viridis", legend=False)
    plt.title("XGBoost Global Feature Importance (Gain)", pad=15, fontweight='bold')
    plt.xlabel("Relative Importance (Fractional Contribution to Gain)")
    plt.ylabel("Features")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def run_evaluation_pipeline(model, X_test, y_test, beta=0.00173):
    """
    Orchestrates the evaluation of the model, saves plots and returns performance metrics.
    """
    # Get probabilities from SMOTE-trained model
    y_prob_smote = model.predict_proba(X_test)[:, 1]
    
    # Recalibrate probability due to SMOTE training oversampling
    # P(y=1|x) = (beta * P_smote) / (beta * P_smote + 1 - P_smote)
    y_prob = (beta * y_prob_smote) / np.maximum(1e-7, (beta * y_prob_smote + 1 - y_prob_smote))
    y_prob = np.clip(y_prob, 0.0, 1.0)
    
    y_pred = (y_prob >= 0.50).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Run threshold analysis
    threshold_comparison = analyze_thresholds(y_test, y_prob)
    
    # Save curves and plots
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(config.FIGURES_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob, save_path=os.path.join(config.FIGURES_DIR, "roc_curve.png"))
    plot_precision_recall_curve(y_test, y_prob, save_path=os.path.join(config.FIGURES_DIR, "precision_recall_curve.png"))
    plot_calibration_curve_plot(y_test, y_prob, save_path=os.path.join(config.FIGURES_DIR, "calibration_curve.png"))
    plot_xgb_feature_importance(model, X_test.columns.tolist(), save_path=os.path.join(config.FIGURES_DIR, "xgboost_feature_importance.png"))
    
    print("\n--- Final Model Metrics (Threshold = 0.50) ---")
    for metric_name, val in metrics.items():
        print(f" - {metric_name}: {val:.4f}")
        
    return metrics, threshold_comparison
