import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import urllib.parse
import requests
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

#####################################
# Common Chemical Name Offline Cache
#####################################
OFFLINE_COMPOUNDS = {
    "glucose": {
        "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O",
        "formula": "C6H12O6",
        "iupac": "(3R,4S,5S,6R)-6-(hydroxymethyl)oxane-2,3,4,5-tetrol",
        "description": "Essential simple sugar (monosaccharide); highly water soluble."
    },
    "aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "formula": "C9H8O4",
        "iupac": "2-acetyloxybenzoic acid",
        "description": "Common pain reliever and anti-inflammatory."
    },
    "caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "formula": "C8H10N4O2",
        "iupac": "1,3,7-trimethylpurine-2,6-dione",
        "description": "Central nervous system stimulant found in coffee and tea."
    },
    "paracetamol": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "formula": "C8H9NO2",
        "iupac": "N-(4-hydroxyphenyl)acetamide",
        "description": "Widely used analgesic and antipyretic (acetaminophen)."
    },
    "acetaminophen": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "formula": "C8H9NO2",
        "iupac": "N-(4-hydroxyphenyl)acetamide",
        "description": "Widely used analgesic and antipyretic (paracetamol)."
    },
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "formula": "C13H18O2",
        "iupac": "2-[4-(2-methylpropyl)phenyl]propanoic acid",
        "description": "NSAID medication for pain and inflammation; poorly water soluble."
    },
    "ethanol": {
        "smiles": "CCO",
        "formula": "C2H6O",
        "iupac": "ethanol",
        "description": "Grain alcohol, miscible with water."
    },
    "sucrose": {
        "smiles": "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
        "formula": "C12H22O11",
        "iupac": "(2R,3R,4S,5S,6R)-2-[(2S,3S,4S,5R)-3,4-dihydroxy-2,5-bis(hydroxymethyl)oxolan-2-yl]oxy-6-(hydroxymethyl)oxane-3,4,5-triol",
        "description": "Table sugar; very highly soluble in water."
    },
    "vitamin c": {
        "smiles": "C(C(C1C(=O)C(=C(O1)O)O)O)O",
        "formula": "C6H8O6",
        "iupac": "(5R)-5-[(1S)-1,2-dihydroxyethyl]-3,4-dihydroxy-5H-furan-2-one",
        "description": "Ascorbic acid, essential nutrient and water-soluble antioxidant."
    },
    "cholesterol": {
        "smiles": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
        "formula": "C27H46O",
        "iupac": "(3S,8S,9S,10R,13R,14S,17R)-10,13-dimethyl-17-[(2R)-6-methylheptan-2-yl]-2,3,4,7,8,9,11,12,14,15,16,17-dodecahydro-1H-cyclopenta[a]phenanthren-3-ol",
        "description": "Sterol lipid; virtually insoluble in water."
    },
    "benzene": {
        "smiles": "c1ccccc1",
        "formula": "C6H6",
        "iupac": "benzene",
        "description": "Aromatic hydrocarbon; sparingly soluble in water."
    },
    "penicillin": {
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",
        "formula": "C16H18N2O4S",
        "iupac": "(2S,5R,6R)-3,3-dimethyl-7-oxo-6-[(2-phenylacetyl)amino]-4-thia-1-azabicyclo[3.2.0]heptane-2-carboxylic acid",
        "description": "Classic beta-lactam antibiotic."
    }
}

@st.cache_data(show_spinner=False)
def translate_name_to_smiles(name):
    """
    Translates an English chemical/common name (e.g. 'glucose', 'aspirin')
    into chemical SMILES, molecular formula, and IUPAC name via PubChem API,
    with local offline fallback.
    """
    cleaned_name = name.strip().lower()
    if not cleaned_name:
        return None

    # 1. Check local cache
    if cleaned_name in OFFLINE_COMPOUNDS:
        data = OFFLINE_COMPOUNDS[cleaned_name]
        return {
            "query": name,
            "smiles": data["smiles"],
            "formula": data["formula"],
            "iupac": data["iupac"],
            "description": data.get("description", ""),
            "source": "Local Chemical Registry"
        }

    # 2. Query PubChem REST API
    try:
        encoded_name = urllib.parse.quote(cleaned_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES,ConnectivitySMILES,MolecularFormula,IUPACName/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            json_data = response.json()
            props = json_data.get("PropertyTable", {}).get("Properties", [])
            if props:
                prop = props[0]
                smiles = prop.get("CanonicalSMILES") or prop.get("ConnectivitySMILES")
                formula = prop.get("MolecularFormula", "N/A")
                iupac = prop.get("IUPACName", "N/A")
                if smiles:
                    return {
                        "query": name,
                        "smiles": smiles,
                        "formula": formula,
                        "iupac": iupac,
                        "description": "Resolved from PubChem (NCBI)",
                        "source": "PubChem API"
                    }
    except Exception:
        pass

    return None

######################
# Delaney Descriptors
######################
def AromaticProportion(m):
    """
    Delaney definition: number of aromatic atoms / number of heavy atoms.
    """
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = [1 for i in aromatic_atoms if i]
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    if HeavyAtom == 0:
        return 0.0
    return AromaticAtom / HeavyAtom

def generate_descriptors(smiles_entries):
    """
    Compute Delaney's 4 molecular descriptors:
    1. MolLogP (Octanol-water partition coefficient)
    2. MolWt (Molecular weight)
    3. NumRotatableBonds (Number of rotatable bonds)
    4. AromaticProportion (Aromatic ratio)
    """
    valid_records = []
    invalid_entries = []

    for entry in smiles_entries:
        smi = entry["smiles"].strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_MolWt = Descriptors.MolWt(mol)
            desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
            desc_AromaticProportion = AromaticProportion(mol)

            valid_records.append({
                "DisplayName": entry.get("name", smi),
                "SMILES": Chem.MolToSmiles(mol),
                "Formula": entry.get("formula", ""),
                "IUPAC": entry.get("iupac", ""),
                "MolLogP": desc_MolLogP,
                "MolWt": desc_MolWt,
                "NumRotatableBonds": desc_NumRotatableBonds,
                "AromaticProportion": desc_AromaticProportion,
                "Mol": mol
            })
        else:
            invalid_entries.append(entry.get("name", smi))

    return valid_records, invalid_entries

def explain_solubility(logs):
    """Plain English solubility rating and explanation."""
    if logs > 0:
        return (
            "Highly Soluble (> 1 mol/L)",
            "🟢 Dissolves very easily in water. Typical of polar molecules like simple sugars and low-chain alcohols."
        )
    elif logs > -2:
        return (
            "Very Soluble (0.01 – 1 mol/L)",
            "🟢 Readily dissolves in water with high solubility. Good bioavailability profile."
        )
    elif logs > -4:
        return (
            "Soluble (0.1 – 10 mmol/L)",
            "🟡 Moderately soluble in water. Meets the typical threshold required for active oral drug candidates."
        )
    elif logs > -6:
        return (
            "Moderately / Sparingly Soluble",
            "🟠 Low aqueous solubility. Often requires dissolution enhancers, formulation aids, or salts."
        )
    else:
        return (
            "Poorly Soluble / Insoluble (< 1 µmol/L)",
            "🔴 Extremely hydrophobic / lipophilic. Will not dissolve meaningfully in water (e.g. fats, waxes, cholesterol)."
        )

######################
# Page Header
######################
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, 'solubility-logo.jpg')

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    try:
        st.image(logo, use_container_width=True)
    except TypeError:
        st.image(logo, use_column_width=True)

st.title("🧪 Molecular Solubility Predictor & Chemical Translator")
st.markdown("""
Predict the **Aqueous Solubility (LogS)** of molecules using Machine Learning! 
You can type **everyday chemical names** (like *glucose*, *aspirin*, *caffeine*) or standard **SMILES** chemical codes.
""")

# Load model
model_path = os.path.join(script_dir, 'solubility_model.pkl')
if not os.path.exists(model_path):
    st.error(f"Model file `solubility_model.pkl` not found in `{script_dir}`.")
    st.stop()

with open(model_path, 'rb') as f:
    load_model = pickle.load(f)

######################
# Sidebar Inputs
######################
st.sidebar.header("Input Method")

input_mode = st.sidebar.radio(
    "How would you like to enter molecules?",
    [
        "🔤 By Molecule Name (e.g. Glucose, Aspirin)",
        "🧪 By SMILES Notation (e.g. CC(=O)Oc1ccccc1C(=O)O)",
        "⚡ Quick Example Presets"
    ]
)

molecules_to_process = []

if input_mode == "🔤 By Molecule Name (e.g. Glucose, Aspirin)":
    st.sidebar.markdown("**Translate Common Names into SMILES:**")
    st.sidebar.caption("Type any chemical or drug name (one per line):")
    name_input = st.sidebar.text_area(
        "Molecules:",
        value="glucose\ncaffeine\naspirin\nibuprofen\nvitamin C",
        height=180
    )
    raw_names = [n.strip() for n in name_input.split('\n') if n.strip()]

    # Resolve names
    for name in raw_names:
        res = translate_name_to_smiles(name)
        if res:
            molecules_to_process.append({
                "name": name.capitalize(),
                "smiles": res["smiles"],
                "formula": res.get("formula", ""),
                "iupac": res.get("iupac", ""),
                "source": res.get("source", "")
            })
        else:
            molecules_to_process.append({
                "name": name,
                "smiles": "INVALID",
                "formula": "",
                "iupac": ""
            })

elif input_mode == "🧪 By SMILES Notation (e.g. CC(=O)Oc1ccccc1C(=O)O)":
    st.sidebar.markdown("**Direct SMILES Input (Data Professor Tutorial):**")
    smiles_input = st.sidebar.text_area(
        "SMILES strings (one per line):",
        value="NCCCC\nCCC\nCN",
        height=180
    )
    raw_smiles = [s.strip() for s in smiles_input.split('\n') if s.strip()]
    for i, smi in enumerate(raw_smiles):
        molecules_to_process.append({
            "name": f"Molecule #{i+1}",
            "smiles": smi,
            "formula": "",
            "iupac": ""
        })

else: # Quick Example Presets
    preset = st.sidebar.selectbox(
        "Select a curated compound set:",
        [
            "Sugars & Nutrients (Glucose, Sucrose, Vitamin C)",
            "Everyday Analgesics (Aspirin, Paracetamol, Ibuprofen)",
            "Solubility Extremes (Ethanol vs. Cholesterol)",
            "Data Professor Video Defaults (NCCCC, CCC, CN)"
        ]
    )
    if preset == "Sugars & Nutrients (Glucose, Sucrose, Vitamin C)":
        preset_names = ["glucose", "sucrose", "vitamin c"]
    elif preset == "Everyday Analgesics (Aspirin, Paracetamol, Ibuprofen)":
        preset_names = ["aspirin", "paracetamol", "ibuprofen"]
    elif preset == "Solubility Extremes (Ethanol vs. Cholesterol)":
        preset_names = ["ethanol", "glucose", "aspirin", "cholesterol"]
    else:
        preset_names = None
        molecules_to_process = [
            {"name": "NCCCC", "smiles": "NCCCC", "formula": "", "iupac": ""},
            {"name": "CCC", "smiles": "CCC", "formula": "", "iupac": ""},
            {"name": "CN", "smiles": "CN", "formula": "", "iupac": ""}
        ]

    if preset_names:
        for name in preset_names:
            res = translate_name_to_smiles(name)
            if res:
                molecules_to_process.append({
                    "name": name.capitalize(),
                    "smiles": res["smiles"],
                    "formula": res.get("formula", ""),
                    "iupac": res.get("iupac", ""),
                    "source": res.get("source", "")
                })

######################
# Translation Summary Expander
######################
if input_mode == "🔤 By Molecule Name (e.g. Glucose, Aspirin)" and molecules_to_process:
    with st.expander("🔍 Chemical Name Translator Results (Name ➔ SMILES)", expanded=True):
        st.write("Here is how your plain-English names were translated into chemical structures:")
        trans_rows = []
        for m in molecules_to_process:
            if m["smiles"] != "INVALID":
                trans_rows.append({
                    "Input Name": m["name"],
                    "Molecular Formula": m.get("formula", "N/A"),
                    "Generated SMILES": m["smiles"],
                    "IUPAC Name": m.get("iupac", "N/A"),
                    "Lookup Source": m.get("source", "Registry")
                })
            else:
                trans_rows.append({
                    "Input Name": m["name"],
                    "Molecular Formula": "Not found",
                    "Generated SMILES": "⚠️ Could not resolve name to structure",
                    "IUPAC Name": "N/A",
                    "Lookup Source": "Failed"
                })
        st.dataframe(pd.DataFrame(trans_rows), use_container_width=True)

######################
# Run Prediction Pipeline
######################
valid_molecules = [m for m in molecules_to_process if m["smiles"] != "INVALID"]
failed_names = [m["name"] for m in molecules_to_process if m["smiles"] == "INVALID"]

if failed_names:
    st.warning(f"⚠️ Could not resolve the following molecule(s): **{', '.join(failed_names)}**. Please check spelling or enter SMILES directly.")

if not valid_molecules:
    st.info("👈 Enter molecules in the sidebar to start predicting.")
else:
    with st.spinner("Calculating Delaney descriptors and predicting aqueous solubility..."):
        records, invalid_smiles = generate_descriptors(valid_molecules)

    if invalid_smiles:
        st.warning(f"⚠️ Invalid chemical SMILES encountered: {', '.join(invalid_smiles)}")

    if records:
        feature_matrix = pd.DataFrame([
            [r["MolLogP"], r["MolWt"], r["NumRotatableBonds"], r["AromaticProportion"]]
            for r in records
        ], columns=["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"])

        preds = load_model.predict(feature_matrix)

        for i, r in enumerate(records):
            r["PredictedLogS"] = round(float(preds[i]), 3)
            classification, explanation = explain_solubility(preds[i])
            r["Classification"] = classification
            r["Explanation"] = explanation

        # Display results in 3 tabs
        tab1, tab2, tab3 = st.tabs(["📊 Prediction Overview", "🧬 2D Structures & Details", "📐 ESOL Model Mathematics"])

        with tab1:
            st.subheader("Solubility Predictions Summary")

            display_df = pd.DataFrame([{
                "Compound": r["DisplayName"],
                "Formula": r["Formula"] if r["Formula"] else "—",
                "LogS": r["PredictedLogS"],
                "Solubility Class": r["Classification"],
                "MolLogP (Lipophilicity)": round(r["MolLogP"], 2),
                "MolWt (g/mol)": round(r["MolWt"], 2),
                "Rotatable Bonds": r["NumRotatableBonds"],
                "Aromatic Ratio": round(r["AromaticProportion"], 2),
                "Canonical SMILES": r["SMILES"]
            } for r in records])

            st.dataframe(display_df, use_container_width=True)

            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="solubility_predictions.csv",
                mime="text/csv"
            )

        with tab2:
            st.subheader("2D Chemical Structures & Solubility Assessment")
            cols = st.columns(min(len(records), 3))
            for i, r in enumerate(records):
                col = cols[i % len(cols)]
                with col:
                    img = Draw.MolToImage(r["Mol"], size=(300, 240))
                    st.image(img, use_container_width=True)
                    st.markdown(f"### {r['DisplayName']}")
                    if r["Formula"]:
                        st.caption(f"Formula: `{r['Formula']}`")
                    st.metric("Predicted LogS", f"{r['PredictedLogS']} mol/L")
                    st.info(f"**{r['Classification']}**\n\n{r['Explanation']}")

        with tab3:
            st.subheader("How Delaney's ESOL Model Works")
            st.markdown(r"""
            **Aqueous Solubility ($\text{LogS}$)** is the $\log_{10}$ of solubility measured in $\text{mol/L}$.
            
            Delaney discovered that 4 core structural properties dictate how well an organic molecule dissolves in water:
            1. **$\text{MolLogP}$ (Lipophilicity)**: Higher values indicate the molecule prefers fat/oil over water, significantly decreasing solubility.
            2. **$\text{MolWt}$ (Molecular Weight)**: Larger molecules require more energy to create a cavity in the water solvent lattice.
            3. **$\text{NumRotatableBonds}$**: Flexible molecules have higher conformational entropy.
            4. **$\text{AromaticProportion}$**: Rigid aromatic rings (like benzene) pack tightly in crystalline state and reduce solubility in water.

            ### Model Equation:
            $$\text{LogS} = \text{Intercept} + c_1(\text{MolLogP}) + c_2(\text{MolWt}) + c_3(\text{NumRotatableBonds}) + c_4(\text{AromaticProportion})$$

            ### Coefficients in your trained model:
            """)
            coef_table = pd.DataFrame({
                "Descriptor": ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"],
                "Learned Coefficient": load_model.coef_
            })
            st.table(coef_table)
            st.write(f"**Model Intercept:** `{load_model.intercept_:.4f}`")

st.markdown("---")
st.caption("Bioinformatics Web App | Molecular Solubility Predictor with Chemical Translator | Built with Streamlit & RDKit")
