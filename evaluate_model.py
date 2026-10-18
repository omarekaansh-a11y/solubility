import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from scipy.stats import pearsonr

# Set directory paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "delaney_dataset.csv")
model_path = os.path.join(script_dir, "solubility_model.pkl")
artifact_dir = r"C:\Users\Ashutosh\.gemini\antigravity\brain\e72cd854-ab53-40b9-b61c-16237b207b03"

# 1. Load Model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 2. Load Dataset
df = pd.read_csv(dataset_path)
print(f"Loaded Delaney dataset with {len(df)} compounds.")

features = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
X = df[features]
y_true = df["logS"]

# 3. Model Predictions
y_pred = model.predict(X)

# 4. Metrics Calculation
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
med_ae = median_absolute_error(y_true, y_pred)
pearson_corr, _ = pearsonr(y_true, y_pred)

residuals = y_true - y_pred
within_05 = np.mean(np.abs(residuals) <= 0.5) * 100
within_10 = np.mean(np.abs(residuals) <= 1.0) * 100

print("\n" + "="*50)
print("       MODEL EVALUATION REPORT (1,144 Compounds)")
print("="*50)
print(f"R² Score (Coefficient of Determination) : {r2:.4f}")
print(f"Pearson Correlation (r)                 : {pearson_corr:.4f}")
print(f"Root Mean Squared Error (RMSE)          : {rmse:.4f} log units")
print(f"Mean Absolute Error (MAE)               : {mae:.4f} log units")
print(f"Median Absolute Error                   : {med_ae:.4f} log units")
print(f"Predictions within ±0.5 LogS            : {within_05:.1f}%")
print(f"Predictions within ±1.0 LogS            : {within_10:.1f}%")
print("="*50)

# 5. Breakdown by Solubility Range
ranges = [
    ("Highly Soluble (logS > 0)", y_true > 0),
    ("Soluble (-2 < logS <= 0)", (y_true > -2) & (y_true <= 0)),
    ("Moderately Soluble (-4 < logS <= -2)", (y_true > -4) & (y_true <= -2)),
    ("Poorly Soluble (-6 < logS <= -4)", (y_true > -6) & (y_true <= -4)),
    ("Extremely Insoluble (logS <= -6)", y_true <= -6),
]

print("\nAccuracy Breakdown by Solubility Category:")
print(f"{'Category':<35} | {'Count':<6} | {'MAE':<8} | {'RMSE':<8}")
print("-" * 65)
for label, mask in ranges:
    sub_count = np.sum(mask)
    if sub_count > 0:
        sub_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        sub_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        print(f"{label:<35} | {sub_count:<6} | {sub_mae:<8.3f} | {sub_rmse:<8.3f}")

# 6. Generate Publication-Quality Evaluation Plot
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)

# Scatter: True vs Predicted
ax1.scatter(y_true, y_pred, alpha=0.55, edgecolors='none', color='#38bdf8', s=35, label='Delaney Compounds (n=1144)')
min_val = min(y_true.min(), y_pred.min()) - 0.5
max_val = max(y_true.max(), y_pred.max()) + 0.5
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Agreement (y = x)')

# Linear trendline
z = np.polyfit(y_true, y_pred, 1)
p = np.poly1d(z)
ax1.plot(np.sort(y_true), p(np.sort(y_true)), color='#f59e0b', lw=1.8, linestyle=':', label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

ax1.set_title(f"Experimental vs. Predicted LogS\n$R^2 = {r2:.3f}$ | $RMSE = {rmse:.3f}$ | $r = {pearson_corr:.3f}$", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Experimental LogS (mol/L)", fontsize=11)
ax1.set_ylabel("Model Predicted LogS (mol/L)", fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.25)
ax1.legend(loc='upper left', framealpha=0.8)
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)

# Residual Distribution
ax2.hist(residuals, bins=35, color='#818cf8', edgecolor='#1e1b4b', alpha=0.85)
ax2.axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
ax2.axvline(np.mean(residuals), color='#f59e0b', linestyle=':', lw=2, label=f'Mean Error = {np.mean(residuals):.3f}')
ax2.set_title(f"Residual Distribution (Prediction Errors)\nMAE = {mae:.3f} | {within_10:.1f}% within ±1.0 LogS", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Residual (Experimental - Predicted LogS)", fontsize=11)
ax2.set_ylabel("Compound Count", fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.25)
ax2.legend(loc='upper right', framealpha=0.8)

plt.tight_layout()

# Save plot to both project directory and artifact brain folder
plot_proj = os.path.join(script_dir, "model_accuracy_evaluation.png")
plot_art = os.path.join(artifact_dir, "model_accuracy_evaluation.png")
fig.savefig(plot_proj, bbox_inches='tight')
fig.savefig(plot_art, bbox_inches='tight')
print(f"\nSaved evaluation plot to:\n1. {plot_proj}\n2. {plot_art}")
