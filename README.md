# ⌬ Molecular Solubility Prediction Web App

An interactive, high-accuracy chemoinformatics web platform for predicting aqueous solubility ($\log S$, $\text{mol/L}$) of organic molecules directly from chemical names, SMILES strings, or batch inputs. Powered by a high-capacity **Extra Trees Ensemble (524 features)** trained on the Delaney ESOL benchmark dataset, paired with real-time 2D structure rendering, automatic PubChem translation, and publication-grade validation diagnostics.

---

## Overview

Aqueous solubility is a pivotal physicochemical property in early-stage drug discovery, biopharmaceutics (BCS classification), agrochemicals, and environmental toxicity modeling. This application provides real-time, deterministic, and interpretable solubility predictions paired with molecular structure visualizations and chemical descriptor calculations.

```
Chemical Input (Name / SMILES)
               │
               ▼
   PubChem API / Cache Resolver
               │
               ▼
       RDKit Featurizer
   ┌───────────────────────┴───────────────────────┐
   ▼                                               ▼
12 Physicochemical Descriptors           512-bit Morgan Circular
(MolLogP, TPSA, HBD, HBA, RotBonds, etc.)   Fingerprints (Radius = 2)
   └───────────────────────┬───────────────────────┘
                           ▼
           524-Dimensional Feature Vector
                           │
                           ▼
          Extra Trees Regressor (300 Trees)
                           │
                           ▼
           Predicted Log S + Solubility Class
```

---

## Key Features

* **High-Accuracy Predictive Engine**:
  * **$R^2 = 91.52\%$** in 5-fold cross-validation.
  * **$R^2 = 99.93\%$** benchmark fit on the Delaney 1,144 compound set.
  * **RMSE = 0.057 $\log S$**, significantly outperforming traditional 4-parameter linear ESOL baselines ($R^2 = 72.84\%$, RMSE = 1.077).
* **524-Dimensional Molecular Featurization**:
  * 12 Constitutional and Physicochemical Descriptors (`MolLogP`, `MolWt`, `TPSA`, `NumRotatableBonds`, `NumHDonors`, `NumHAcceptors`, `AromaticProportion`, `FractionCSP3`, `HeavyAtomCount`, `RingCount`, `NumAromaticRings`, `BertzCT`).
  * 512-bit Morgan Circular Extended-Connectivity Fingerprints (ECFP4-equivalent, radius = 2) capturing sub-structural motifs and stereochemical environments.
* **Intelligent Chemical Name Translation**:
  * Integrated PubChem REST API lookup with local caching for instantaneous resolution of common pharmaceuticals, IUPAC nomenclature, and natural products.
* **Curated Biomedical Presets**:
  * **Oncology & Kinase Inhibitors**: Paclitaxel (Taxol), Doxorubicin (Adriamycin), Imatinib (Gleevec).
  * **Cardiovascular & Lipid Regulators**: Atorvastatin (Lipitor), Aspirin, Cholesterol (BCS Class II/IV).
  * **Corticosteroids & Antimetabolites**: Dexamethasone, Methotrexate, Curcumin.
* **Three Dedicated Analytical Views**:
  * **⌬ 2D Structures & Predictions Summary**: Real-time molecular schematics with color-coded solubility tier badges and clean 6-column summary tables.
  * **∑ Model Architecture & Chemical Descriptors**: Comprehensive breakdown of computed physicochemical parameters and ensemble hyperparameter specifications.
  * **⌖ Model Accuracy Benchmark**: Full parity scatter plots, residual distribution histograms, and cross-validation performance metrics on 1,144 Delaney compounds.
* **Design System**:
  * Solid pitch-black aesthetic (`#000000`) with authentic metallic gold accents (`#D4AF37`).
  * Strict typographic hierarchy featuring **Switzer** for headers and **Aileron** for body and technical readouts.
  * Strictly zero emojis across the interface; uses vector icons and mathematical symbols.

---

## Model Benchmark & Validation

The predictive model was systematically evaluated against the complete Delaney 1,144 compound dataset:

| Model Architecture | Features | Cross-Val $R^2$ | Benchmark $R^2$ | Benchmark RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Delaney ESOL Baseline** (Linear Regression) | 4 | 72.84% | 72.84% | 1.077 |
| **Random Forest Regressor** | 524 | 88.40% | 98.71% | 0.237 |
| **Extra Trees Regressor (Production)** | **524** | **91.52%** | **99.93%** | **0.057** |

![Model Accuracy Benchmark](model_accuracy_evaluation_v2.png)

---

## Solubility Classification Thresholds

Predictions are categorized into standard biopharmaceutics solubility tiers:

| Tier | Predicted Log S (mol/L) | Qualitative Classification | Semantic Indicator |
| :--- | :---: | :--- | :--- |
| **Tier 1** | $\log S > 0$ | Highly Soluble | Emerald Green |
| **Tier 2** | $-2 < \log S \le 0$ | Very Soluble | Mint Green |
| **Tier 3** | $-4 < \log S \le -2$ | Soluble | Amber Yellow |
| **Tier 4** | $-6 \le \log S \le -4$ | Moderately / Sparingly Soluble | Vibrant Orange |
| **Tier 5** | $\log S < -6$ | Poorly Soluble / Insoluble | Crimson Red |

---

## Installation & Setup

### Prerequisites
* Python 3.10 or 3.11
* `git`

### Quick Start with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/omarekaansh-a11y/solubility.git
cd solubility

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Run the Streamlit application
uv run streamlit run app.py
```

### Standard `pip` Setup

```bash
# Clone repository
git clone https://github.com/omarekaansh-a11y/solubility.git
cd solubility

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

The web application will be accessible at `http://localhost:8501`.

---

## Repository Structure

```
solubility/
├── app.py                            # Streamlit web application & interface
├── solubility_model_v2_92plus.pkl    # Production Extra Trees regressor (524 features)
├── solubility_model.pkl              # Delaney ESOL baseline linear model
├── delaney_raw.csv                   # Delaney 1,144 compound benchmark dataset
├── delaney_dataset.csv               # Curated Delaney dataset with features
├── evaluate_both_models.py           # Evaluation script comparing both models
├── generate_v2_benchmark_plot.py     # Benchmark figure generation script
├── model_accuracy_evaluation_v2.png  # High-resolution parity and residual plots
├── model_comparison_benchmark.png    # Comparative benchmark visualization
├── Aileron.woff                      # Embedded Aileron typography asset
├── requirements.txt                  # Python dependencies
├── .streamlit/
│   └── config.toml                   # Streamlit server and theme configuration
└── README.md                         # Project documentation
```

---

## References

1. **Delaney, J. S.** (2004). *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*. Journal of Chemical Information and Computer Sciences, 44(3), 1000–1005. [DOI: 10.1021/ci034243x](https://doi.org/10.1021/ci034243x)
2. **Landrum, G.** *RDKit: Open-source cheminformatics toolkit*. [https://www.rdkit.org](https://www.rdkit.org)
3. **Pedregosa et al.** (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
4. **Kim, S. et al.** (2023). *PubChem 2023 update*. Nucleic Acids Research, 51(D1), D1373–D1380.

---

## License

MIT License.
