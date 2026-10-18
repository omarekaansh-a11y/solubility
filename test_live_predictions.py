import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd

# Load models
linear_model = pickle.load(open('solubility_model.pkl', 'rb'))
v2_meta = pickle.load(open('solubility_model_v2_92plus.pkl', 'rb'))
v2_model = v2_meta['model']

test_compounds = {
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Sucrose": "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
}

print(f"{'Compound':<15} | {'Delaney Linear LogS':<20} | {'High-Accuracy Extra Trees LogS':<30}")
print("-" * 70)

for name, smi in test_compounds.items():
    mol = Chem.MolFromSmiles(smi)
    heavy = Descriptors.HeavyAtomCount(mol)
    aromatic = sum([1 for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()])
    ar_prop = aromatic / heavy if heavy > 0 else 0.0
    
    # 4 Linear features
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    X_lin = pd.DataFrame([[logp, mw, rb, ar_prop]], columns=["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"])
    pred_lin = linear_model.predict(X_lin)[0]
    
    # 12 v2 features + 512 Morgan bits
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    mr = Descriptors.MolMR(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    n_aro_r = Descriptors.NumAromaticRings(mol)
    n_ali_r = Descriptors.NumAliphaticRings(mol)
    n_rings = Descriptors.RingCount(mol)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    fp_arr = np.zeros((512,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    
    v2_vec = np.hstack([
        np.array([logp, mw, rb, ar_prop, tpsa, hbd, hba, mr, fsp3, n_aro_r, n_ali_r, n_rings]),
        fp_arr
    ]).reshape(1, -1)
    pred_v2 = v2_model.predict(v2_vec)[0]
    
    print(f"{name:<15} | {pred_lin:18.3f} | {pred_v2:28.3f}")
