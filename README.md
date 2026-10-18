# 🧪 Bioinformatics Web App: Molecular Solubility Predictor

This repository contains the interactive Streamlit web application to predict aqueous solubility (LogS) of molecules from chemical SMILES structures, based on **Data Professor's Streamlit Tutorial #7**.

## 📌 Features
- **Descriptor Calculation**: Automatically computes 4 molecular descriptors using **RDKit**:
  - `MolLogP` (Octanol-water partition coefficient)
  - `MolWt` (Molecular weight)
  - `NumRotatableBonds` (Number of rotatable bonds)
  - `AromaticProportion` (Proportion of aromatic heavy atoms)
- **Machine Learning Inference**: Uses Delaney's linear regression model (`solubility_model.pkl`).
- **Chemical Structure Visualization**: Generates 2D molecular structure diagrams in real-time.
- **Batch Processing & Export**: Predicts solubility for single molecules or large batches with 1-click CSV download.
- **Reference**: John S. Delaney. *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*. J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000–1005.

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit web app
streamlit run app.py
```

## 🌐 Deploying to Streamlit Cloud
1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository and select `app.py`.
4. Click **Deploy**!
