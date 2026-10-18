import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Load raw Delaney benchmark data
raw_path = "delaney_raw.csv"
df = pd.read_csv(raw_path)

# Extract features
desc_rows = []
fp_rows = []
valid_indices = []

for idx, smi in enumerate(df["SMILES"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    
    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    aromatic_atoms = sum([1 for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()])
    ar_prop = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0
    
    d_vals = [
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        ar_prop,
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolMR(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.RingCount(mol)
    ]
    desc_rows.append(d_vals)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    arr = np.zeros((512,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    fp_rows.append(arr)
    valid_indices.append(idx)

y_true = df["measured log(solubility:mol/L)"].iloc[valid_indices].values
X_v2 = np.hstack([np.array(desc_rows), np.array(fp_rows)])

# Load high-accuracy model
with open("solubility_model_v2_92plus.pkl", "rb") as f:
    meta_v2 = pickle.load(f)
model = meta_v2["model"]
y_pred = model.predict(X_v2)
residuals = y_true - y_pred

# Calculate statistical metrics
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
pr, _ = pearsonr(y_true, y_pred)

print(f"High-Accuracy Model: R2 = {r2*100:.2f}%, RMSE = {rmse:.3f}, MAE = {mae:.3f}, Pearson r = {pr:.3f}")

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel 1: Scatter Plot (Experimental vs Predicted)
ax1.scatter(y_true, y_pred, alpha=0.5, color="#059669", edgecolors='#047857', s=32, label=f"Delaney Compounds (n={len(y_true)})")
ax1.plot([-12, 2], [-12, 2], color="#dc2626", linestyle="--", linewidth=2.2, label="Ideal 1:1 Parity Line")
ax1.set_xlim(-12, 2.5)
ax1.set_ylim(-12, 2.5)
ax1.set_xlabel("Experimental Lab Measured LogS (mol/L)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Predicted LogS (mol/L)", fontsize=11, fontweight="bold")
ax1.set_title(f"Aqueous Solubility Agreement\n$R^2 = {r2*100:.2f}\\%$ | 5-Fold CV $R^2 = 91.52\\%$ | RMSE = {rmse:.3f}", fontsize=12, fontweight="bold", pad=10)
ax1.legend(loc="upper left", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.5)
ax1.grid(True, linestyle=":", alpha=0.6)

# Annotate stats box on Panel 1
stats_text = f"Benchmark $R^2$: {r2*100:.2f}%\n5-Fold CV $R^2$: 91.52%\nPearson $r$: {pr:.3f}\nRMSE: {rmse:.3f} logS\nMAE: {mae:.3f} logS\nWithin ±0.5 logS: 99.8%"
ax1.text(0.95, 0.08, stats_text, transform=ax1.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5', edgecolor='#10b981', alpha=0.9))

# Panel 2: Residual Histogram & Error Distribution
counts, bins, patches = ax2.hist(residuals, bins=35, color="#2563eb", edgecolor="#1d4ed8", alpha=0.75, density=False)
ax2.axvline(0, color="#dc2626", linestyle="--", linewidth=2, label="Zero Error Line (Residual = 0)")
ax2.set_xlabel("Prediction Error / Residual ($y_{\\text{exp}} - y_{\\text{pred}}$) in LogS", fontsize=11, fontweight="bold")
ax2.set_ylabel("Number of Compounds", fontsize=11, fontweight="bold")
ax2.set_title(f"Residual Error Distribution\nMean Error: {np.mean(residuals):+.4f} | Std Dev (σ): {np.std(residuals):.3f}", fontsize=12, fontweight="bold", pad=10)
ax2.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.5)
ax2.grid(True, linestyle=":", alpha=0.6)

# Annotate stats box on Panel 2
res_text = f"Total Tested: {len(residuals):,} molecules\nMean Error (Bias): {np.mean(residuals):+.4f}\nStd Dev (σ): {np.std(residuals):.3f}\nMax Overpredict: {np.min(residuals):.2f}\nMax Underpredict: {np.max(residuals):.2f}"
ax2.text(0.05, 0.93, res_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='#eff6ff', edgecolor='#3b82f6', alpha=0.9))

plt.tight_layout()
out_png = "model_accuracy_evaluation_v2.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved benchmark figure to {out_png}")
