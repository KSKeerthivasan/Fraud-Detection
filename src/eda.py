import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def perform_eda():
    """Performs EDA on the credit card dataset and saves publication-quality figures."""
    print("Starting Exploratory Data Analysis...")
    
    # Load dataset
    if not os.path.exists(config.RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    # Set styling
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # 1. Class Distribution Analysis
    print("Analyzing class distribution...")
    class_counts = df['Class'].value_counts()
    total_tx = len(df)
    fraud_pct = (class_counts[1] / total_tx) * 100
    
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1f77b4', '#d62728'] # Classic blue and crimson red
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors, hue=class_counts.index, legend=False, ax=ax)
    ax.set_title("Transaction Class Distribution", pad=15, fontweight='bold')
    ax.set_xlabel("Class (0: Legitimate, 1: Fraudulent)")
    ax.set_ylabel("Count (Log Scale)")
    ax.set_yscale('log') # Log scale because of extreme imbalance
    
    # Add count and percentage labels on top of the bars
    for i, count in enumerate(class_counts.values):
        pct = (count / total_tx) * 100
        ax.text(i, count * 1.5, f"{count:,}\n({pct:.3f}%)", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    fig_path = os.path.join(config.FIGURES_DIR, "class_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to {fig_path}")
    print(f"Total Transactions: {total_tx:,} | Legitimate: {class_counts[0]:,} | Fraudulent: {class_counts[1]:,} ({fraud_pct:.4f}%)")
    
    # 2. Amount Distribution Analysis (Log scale)
    print("Analyzing transaction amounts...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We apply np.log1p to the transaction amount to handle the heavy right skewness and $0.00 values safely
    df_copy = df.copy()
    df_copy['Log_Amount'] = np.log1p(df_copy['Amount'])
    
    sns.kdeplot(data=df_copy[df_copy['Class'] == 0], x='Log_Amount', fill=True, color='#1f77b4', label='Legitimate', alpha=0.5, ax=ax)
    sns.kdeplot(data=df_copy[df_copy['Class'] == 1], x='Log_Amount', fill=True, color='#d62728', label='Fraudulent', alpha=0.5, ax=ax)
    
    ax.set_title("Density Distribution of Transaction Amounts (Log Scale)", pad=15, fontweight='bold')
    ax.set_xlabel("Log(Amount + 1) in USD")
    ax.set_ylabel("Density")
    ax.legend()
    
    # Statistical summaries
    fraud_desc = df[df['Class'] == 1]['Amount'].describe()
    legit_desc = df[df['Class'] == 0]['Amount'].describe()
    print(f"Legitimate Tx Amount - Mean: ${legit_desc['mean']:.2f}, Median: ${legit_desc['50%']:.2f}, Max: ${legit_desc['max']:.2f}")
    print(f"Fraudulent Tx Amount - Mean: ${fraud_desc['mean']:.2f}, Median: ${fraud_desc['50%']:.2f}, Max: ${fraud_desc['max']:.2f}")
    
    plt.tight_layout()
    fig_path = os.path.join(config.FIGURES_DIR, "amount_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved amount distribution plot to {fig_path}")
    
    # 3. Correlation Analysis
    print("Computing feature correlations...")
    # Because V1-V28 are orthogonal PCA components, they have zero correlation with each other.
    # But they do have correlation with 'Class', and so do 'Time' and 'Amount'.
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    # Using a coolwarm diverging colormap
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, 
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Feature Correlation Matrix Heatmap", pad=20, fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    fig_path = os.path.join(config.FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved correlation matrix plot to {fig_path}")
    
    # Let's print the top 5 positively and negatively correlated features with Class
    class_corr = corr['Class'].sort_values()
    print("\nTop 5 features negatively correlated with Class (Fraud):")
    print(class_corr.head(5))
    print("\nTop 5 features positively correlated with Class (Fraud):")
    print(class_corr.tail(6).iloc[:-1]) # exclude Class itself
    
    print("EDA completed successfully.")

if __name__ == "__main__":
    perform_eda()
