# 🧪 Molecular Solubility Predictor & Chemical Translator

An interactive bioinformatics web application to predict aqueous solubility (LogS) of organic molecules from chemical structures, featuring an automated Chemical Name Translator (PubChem API) and machine learning predictions.

## 📌 Features
- **Chemical Name Translator**: Type plain English chemical/drug names (e.g., Glucose, Aspirin, Caffeine, Paracetamol) to automatically resolve molecular formula, IUPAC name, and chemical SMILES.
- **Delaney Molecular Descriptors (RDKit)**:
  - `MolLogP` (Octanol-water partition coefficient)
  - `MolWt` (Molecular weight)
  - `NumRotatableBonds` (Number of rotatable bonds)
  - `AromaticProportion` (Proportion of aromatic heavy atoms)
- **Machine Learning Inference**: Pre-trained Delaney ESOL Linear Regression model (`solubility_model.pkl`).
- **2D Chemical Structure Diagrams**: Real-time rendering of molecular structures.
- **Theme Modes**: Modern dark mode by default with instant light mode toggle.
- **Batch Export**: Download predictions and computed descriptors as CSV.
- **Reference**: John S. Delaney. *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*. J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000–1005.

## 🚀 Running the Web App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python -m streamlit run app.py
```
