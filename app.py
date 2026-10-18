import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

# Set page configuration
st.set_page_config(
    page_title="Molecular Solubility Predictor",
    page_icon="🧪",
    layout="wide"
)

######################
# Custom functions
######################
def AromaticProportion(m):
    """
    Calculate the proportion of aromatic atoms to heavy atoms (Delaney definition).
    """
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = [1 for i in aromatic_atoms if i]
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    if HeavyAtom == 0:
        return 0.0
    return AromaticAtom / HeavyAtom

def generate(smiles_list):
    """
    Compute Delaney's 4 molecular descriptors:
    1. MolLogP (Octanol-water partition coefficient)
    2. MolWt (Molecular weight)
    3. NumRotatableBonds (Number of rotatable bonds)
    4. AromaticProportion (Aromatic ratio)
    """
    valid_mols = []
    valid_indices = []
    invalid_smiles = []

    for idx, smi in enumerate(smiles_list):
        smi_clean = smi.strip()
        if not smi_clean:
            continue
        mol = Chem.MolFromSmiles(smi_clean)
        if mol is not None:
            valid_mols.append(mol)
            valid_indices.append(idx)
        else:
            invalid_smiles.append(smi_clean)

    if not valid_mols:
        return None, None, invalid_smiles

    baseData = []
    for mol in valid_mols:
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        baseData.append([
            desc_MolLogP,
            desc_MolWt,
            desc_NumRotatableBonds,
            desc_AromaticProportion
        ])

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors, valid_mols, invalid_smiles

def classify_solubility(logs_value):
    """Qualitative classification of LogS (mol/L)"""
    if logs_value > 0:
        return "Highly soluble"
    elif logs_value > -2:
        return "Very soluble"
    elif logs_value > -4:
        return "Soluble"
    elif logs_value > -6:
        return "Moderately soluble"
    else:
        return "Poorly / Sparingly soluble"

######################
# Page Title & Header
######################
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, 'solubility-logo.jpg')

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    try:
        st.image(logo, use_container_width=True)
    except TypeError:
        st.image(logo, use_column_width=True)

st.title("🧪 Molecular Solubility Prediction Web App")
st.markdown("""
This bioinformatics application predicts the **Aqueous Solubility (LogS)** of chemical molecules using Machine Learning.

* **Reference:** John S. Delaney. *[ESOL: Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x)*. **J. Chem. Inf. Comput. Sci.** 2004, 44, 3, 1000–1005.
* **Model:** Linear Regression trained on Delaney's ESOL dataset (Molecular LogP, Molecular Weight, Rotatable Bonds, and Aromatic Proportion).
---
""")

######################
# Sidebar - User Input
######################
st.sidebar.header("Input Chemical Molecules")

# Example presets
preset_choice = st.sidebar.selectbox(
    "Choose Preset or Custom:",
    [
        "Data Professor Default (3 Molecules)",
        "Famous Pharmaceuticals (Aspirin, Caffeine, Paracetamol, Ibuprofen)",
        "Custom Input"
    ]
)

if preset_choice == "Data Professor Default (3 Molecules)":
    default_smiles = "NCCCC\nCCC\nCN"
elif preset_choice == "Famous Pharmaceuticals (Aspirin, Caffeine, Paracetamol, Ibuprofen)":
    default_smiles = "CC(=O)Oc1ccccc1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCC(=O)Nc1ccc(O)cc1\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
else:
    default_smiles = "NCCCC\nCCC\nCN"

smiles_input = st.sidebar.text_area(
    "Enter SMILES string(s) (one per line):",
    value=default_smiles,
    height=180
)

# Load trained model
model_path = os.path.join(script_dir, 'solubility_model.pkl')
if not os.path.exists(model_path):
    st.error(f"Trained model `solubility_model.pkl` not found in `{script_dir}`. Please make sure the model file is present.")
    st.stop()

with open(model_path, 'rb') as f:
    load_model = pickle.load(f)

######################
# Process SMILES & Predict
######################
raw_smiles = [s.strip() for s in smiles_input.split('\n') if s.strip()]

if not raw_smiles:
    st.info("👈 Please enter one or more SMILES strings in the sidebar to predict solubility.")
else:
    with st.spinner("Computing Delaney descriptors and running model predictions..."):
        X, valid_mols, invalid_smiles = generate(raw_smiles)

    if invalid_smiles:
        st.warning(f"⚠️ Could not parse the following SMILES string(s): {', '.join(invalid_smiles)}")

    if X is not None and not X.empty:
        # Run predictions
        predictions = load_model.predict(X)

        # Prepare summary dataframe
        results_df = X.copy()
        results_df.insert(0, "SMILES", [Chem.MolToSmiles(m) for m in valid_mols])
        results_df["Predicted LogS"] = np.round(predictions, 3)
        results_df["Solubility Class"] = [classify_solubility(p) for p in predictions]

        # Display tabs
        tab1, tab2, tab3 = st.tabs(["📊 Prediction Results", "🧬 2D Molecular Structures", "ℹ️ Delaney Formula & Details"])

        with tab1:
            st.subheader("Summary Table")
            st.dataframe(results_df, use_container_width=True)

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="solubility_predictions.csv",
                mime="text/csv",
            )

        with tab2:
            st.subheader("2D Chemical Structures")
            cols = st.columns(min(len(valid_mols), 4))
            for i, mol in enumerate(valid_mols):
                col = cols[i % len(cols)]
                with col:
                    img = Draw.MolToImage(mol, size=(280, 240))
                    st.image(img, caption=f"Molecule #{i+1}: {Chem.MolToSmiles(mol)}", use_container_width=True)
                    st.metric(label="Predicted LogS", value=f"{predictions[i]:.2f}")
                    st.caption(f"Class: **{classify_solubility(predictions[i])}**")

        with tab3:
            st.subheader("Delaney ESOL Methodology")
            st.markdown(r"""
            The Delaney ESOL model calculates solubility using the linear relationship:
            
            $$\text{LogS} = 0.16 - 0.63 \times \text{MolLogP} - 0.0062 \times \text{MolWt} + 0.066 \times \text{NumRotatableBonds} - 0.74 \times \text{AromaticProportion}$$

            ### Model Parameters in Your Trained Model:
            """)
            coef_df = pd.DataFrame({
                "Descriptor": ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"],
                "Model Coefficient": load_model.coef_
            })
            st.table(coef_df)
            st.write(f"**Model Intercept:** `{load_model.intercept_:.4f}`")

st.markdown("---")
st.caption("Bioinformatics Web App | Tutorial by Data Professor (Chanin Nantasenamat) | Built with Streamlit & RDKit")
