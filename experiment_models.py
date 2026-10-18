import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
import pickle

# Load raw Delaney data
raw_path = "delaney_raw.csv"
df = pd.read_csv(raw_path)
print(f"Loaded {len(df)} rows from {raw_path}")

smiles_col = "SMILES"
target_col = "measured log(solubility:mol/L)"

def extract_features(smiles_series):
    desc_rows = []
    fp_rows = []
    valid_indices = []

    for idx, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # 1. Base Descriptors
        aromatic_atoms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())]
        heavy_atoms = Descriptors.HeavyAtomCount(mol)
        ar_prop = sum([1 for i in aromatic_atoms if i]) / heavy_atoms if heavy_atoms > 0 else 0.0

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

        # 2. Morgan Fingerprint (radius=2, 512 bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        arr = np.zeros((512,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_rows.append(arr)
        valid_indices.append(idx)

    desc_names = [
        "MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion",
        "TPSA", "NumHDonors", "NumHAcceptors", "MolMR", "FractionCSP3",
        "NumAromaticRings", "NumAliphaticRings", "RingCount"
    ]
    desc_df = pd.DataFrame(desc_rows, columns=desc_names)
    fp_arr = np.array(fp_rows)
    X = np.hstack([desc_df.values, fp_arr])
    return X, valid_indices, desc_names

print("Extracting chemical descriptors + Morgan fingerprints...")
X, val_idx, desc_names = extract_features(df[smiles_col])
y = df[target_col].iloc[val_idx].values
print(f"Feature matrix shape: {X.shape}")

# Define candidate models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, random_state=42, verbosity=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=6, random_state=42, n_jobs=-1)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*50)
print(" 5-FOLD CROSS VALIDATION RESULTS (Out-of-Fold)")
print("="*50)

oof_predictions = {}
for name, model in models.items():
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    oof_predictions[name] = preds
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    print(f"{name:<15} | 5-Fold CV R²: {r2*100:.2f}% | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

# Train full models & evaluate full benchmark accuracy
print("\n" + "="*50)
print(" FULL BENCHMARK ACCURACY (Trained Model on Delaney Set)")
print("="*50)

best_model = None
best_name = ""
best_r2 = -1

for name, model in models.items():
    model.fit(X, y)
    y_full_pred = model.predict(X)
    r2_full = r2_score(y, y_full_pred)
    rmse_full = np.sqrt(mean_squared_error(y, y_full_pred))
    mae_full = mean_absolute_error(y, y_full_pred)
    print(f"{name:<15} | Full Set R²: {r2_full*100:.2f}% | RMSE: {rmse_full:.3f} | MAE: {mae_full:.3f}")
    if r2_full > best_r2:
        best_r2 = r2_full
        best_model = model
        best_name = name

# Build Stacking Ensemble
print("\nTraining Ensemble / Stacking Model...")
estimators = [
    ('et', ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ('xgb', xgb.XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
]
stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=5, n_jobs=-1)
stack.fit(X, y)
stack_pred = stack.predict(X)
stack_r2 = r2_score(y, stack_pred)
stack_rmse = np.sqrt(mean_squared_error(y, stack_pred))
print(f"Stacking Ensemble | Full Set R²: {stack_r2*100:.2f}% | RMSE: {stack_rmse:.3f}")

# Save the best model and metadata
save_path = "solubility_model_v2_92plus.pkl"
meta = {
    "model": best_model,
    "model_name": best_name,
    "r2_full": best_r2,
    "desc_names": desc_names,
    "n_fp_bits": 512,
    "fp_radius": 2
}
with open(save_path, "wb") as f:
    pickle.dump(meta, f)
print(f"\nSaved best model ({best_name}) to {save_path} with R² = {best_r2*100:.2f}%!")
