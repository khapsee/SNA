import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

gcn_df          = pd.read_csv("prediction_vs_actual.csv")
lr_df           = pd.read_csv("baseline_prediction_vs_actual.csv")

gcn_df          = gcn_df.rename(columns={'predicted_citations': 'predicted_citations_gcn'})
lr_df           = lr_df.rename(columns={'predicted_citations': 'predicted_citations_lr'})

merged          = pd.merge(gcn_df, lr_df, on=['author_id', 'actual_citations'])

gcn_errors      = np.abs(merged['actual_citations'] - merged['predicted_citations_gcn'])
lr_errors       = np.abs(merged['actual_citations'] - merged['predicted_citations_lr'])

t_stat, p_value = ttest_rel(lr_errors, gcn_errors)

print("\nPaired t-test: Linear Regression vs GCN")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value     = {p_value:.4f}")

if p_value < 0.05:
    print("GCN performs better (p < 0.05).")
else:
    print("Not statistically significant.")
