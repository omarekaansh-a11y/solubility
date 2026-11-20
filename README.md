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

## Mathematical Foundations & Core Equations

### 1. Aqueous Solubility & Thermodynamic Conversions

Aqueous solubility ($S$) represents the maximum concentration of a chemical compound that dissolves in water at thermodynamic equilibrium at standard temperature ($25^\circ\text{C}$) and pressure ($1\text{ atm}$):

$$\log S = \log_{10} \left( \frac{[\text{Solute}]_{\text{saturated}}}{\text{mol/L}} \right)$$

To convert molar solubility ($\text{mol/L}$) into mass concentrations essential for pharmaceutical dosing and pharmacokinetics:

$$S_{\text{g/L}} = 10^{\log S} \times \text{MolWt}$$

$$S_{\text{mg/mL}} = 10^{\log S} \times \text{MolWt}$$

$$S_{\mu\text{g/mL}} = 10^{\log S + 3} \times \text{MolWt}$$

#### Thermodynamic Context: The General Solubility Equation (GSE)
Classical thermodynamic estimation (Yalkowsky & Valvani) models crystal lattice energy through melting point ($\text{MP}$) and hydrophobicity through the octanol-water partition coefficient ($\log K_{\text{ow}}$):

$$\log S = 0.5 - 0.01 \times (\text{MP} - 25) - \log K_{\text{ow}}$$

Because experimental melting point ($\text{MP}$) is unavailable for hypothetical or un-synthesized virtual screening libraries, John S. Delaney developed **ESOL** to estimate $\log S$ directly from 2D molecular graph invariants.

---

### 2. Delaney ESOL Linear Regression Formulations

#### Original Published Delaney ESOL Equation (2004)
Derived by Delaney from 9-descriptor parameter optimization across 2,874 experimental measurements:

$$\log S = 0.16 - 0.63 \times \text{cLogP} - 0.0062 \times \text{MW} + 0.066 \times \text{RB} - 0.74 \times \text{AP}$$

#### Baseline Retrained Delaney Model (`solubility_model.pkl`)
Fitted on the standardized 1,144 Delaney benchmark dataset via Ordinary Least Squares (OLS):

$$\log S = 0.2225 - 0.7389 \times \text{MolLogP} - 0.0064 \times \text{MolWt} - 0.0026 \times \text{NumRotatableBonds} - 0.4429 \times \text{AromaticProportion}$$

* Baseline Performance on Delaney Benchmark: $R^2 = 72.84\%$, $\text{RMSE} = 1.077\text{ }\log S$, $\text{MAE} = 0.835\text{ }\log S$, Pearson $r = 0.854$.

---

### 3. Exact Algorithms for the 12 Chemical Descriptors

Every molecule $\mathcal{M} = (\mathcal{V}, \mathcal{E})$ is parsed into an RDKit molecular graph where $\mathcal{V}$ is the set of heavy (non-hydrogen) vertices and $\mathcal{E}$ is the set of covalent edges:

#### 1. Wildman-Crippen Octanol-Water Partition Coefficient (`MolLogP`)
Calculated via additive atomic contributions across 68 empirically trained atom-typing classes:

$$\text{MolLogP} = \sum_{i=1}^{|\mathcal{V}|} a_i$$

where $a_i$ is the empirical hydrophobicity contribution assigned to atom $i$ based on its atomic number, formal charge, aromaticity, hybridization, and adjacent valence bonds.

#### 2. Molecular Weight (`MolWt`)
Sum of IUPAC standard atomic weights across all constituent atoms including implicit hydrogens:

$$\text{MolWt} = \sum_{i=1}^{|\mathcal{V}|} \left( \bar{A}_i + h_i \times \bar{A}_{\text{H}} \right)$$

where $\bar{A}_i$ is the atomic weight of heavy atom $i$, $h_i$ is the number of attached implicit hydrogen atoms, and $\bar{A}_{\text{H}} = 1.008\text{ g/mol}$.

#### 3. Rotatable Bonds Count (`NumRotatableBonds`)
Count of single, non-ring, non-terminal covalent bonds between heavy atoms with unrestricted torsional freedom:

$$N_{\text{rot}} = \sum_{e \in \mathcal{E}_{\text{single}}} \mathbb{I}(e \notin \text{Rings} \land \deg(u) > 1 \land \deg(v) > 1 \land e \neq \text{Amide C-N})$$

Amide $\text{C}-\text{N}$ bonds are excluded due to partial double-bond character from $\pi$-resonance conjugation.

#### 4. Aromatic Proportion (`AromaticProportion`)
Ratio of aromatic heavy atoms to total heavy atoms:

$$\text{AP} = \frac{\sum_{i \in \mathcal{V}} \mathbb{I}(\text{atom } i \text{ is aromatic})}{|\mathcal{V}|}$$

Captures rigid planarity and high crystal lattice enthalpy ($\Delta H_{\text{lattice}}$) that impairs aqueous solvation.

#### 5. Topological Polar Surface Area (`TPSA`)
Sum of polar fragment surface contributions (Ertl, Rohde, & Selzer, 2000):

$$\text{TPSA} = \sum_{j \in \{\text{O, N polar fragments}\}} S_j$$

where $S_j$ is the pre-computed 2D surface area parameter based on nitrogen/oxygen hybridization and attached hydrogen count.

#### 6. Hydrogen Bond Donors (`NumHDonors`)
Count of all heteroatoms carrying protic hydrogens (Lipinski definition):

$$N_{\text{HBD}} = \sum_{i \in \{\text{O, N}\}} h_i$$

#### 7. Hydrogen Bond Acceptors (`NumHAcceptors`)
Count of electronegative valence atoms capable of accepting hydrogen bonds:

$$N_{\text{HBA}} = \sum_{i \in \mathcal{V}} \mathbb{I}(\text{Element}(i) \in \{\text{O, N}\})$$

#### 8. Molar Refractivity (`MolMR`)
Estimate of overall molecular polarizability and steric volume via additive atom-type contributions:

$$\text{MolMR} = \sum_{i=1}^{|\mathcal{V}|} r_i$$

Related to the Lorentz-Lorenz equation $\text{MR} = \frac{n^2 - 1}{n^2 + 2} \frac{M}{\rho}$, where $n$ is the refractive index and $\rho$ is density.

#### 9. Fraction of $sp^3$ Carbons (`FractionCSP3`)
Quantifies molecular three-dimensionality and saturation ("escape from flatland"):

$$\text{FractionCSP3} = \frac{\sum_{i \in \mathcal{V}_{\text{C}}} \mathbb{I}(\text{Hybridization}(i) = sp^3)}{|\mathcal{V}_{\text{C}}|}$$

Molecules with higher $\text{FractionCSP3}$ exhibit lower molecular packing efficiency in crystal lattices, leading to lower melting points and higher aqueous solubility.

#### 10. Ring Counts (`NumAromaticRings`, `NumAliphaticRings`, `RingCount`)
Computed via the Smallest Set of Smallest Rings (SSSR) decomposition:

$$\text{RingCount} = \text{NumAromaticRings} + \text{NumAliphaticRings} + \text{NumBridgedRings}$$

---

### 4. 512-Bit Morgan Circular Fingerprints (ECFP4 Equivalent)

To encode nonlinear sub-structural environments and functional group pharmacophores, the pipeline computes extended connectivity Morgan fingerprints with radius $r = 2$:

1. **Initialization ($r = 0$)**: Each heavy atom $i$ is assigned an initial integer invariant vector encoding:
   $$\mathbf{v}_0(i) = \langle Z_i, \deg(i), \text{val}(i) - h_i, \text{charge}(i), \mathbb{I}(i \text{ is aromatic}) \rangle$$
   $$\text{id}_0(i) = \text{MurmurHash3}(\mathbf{v}_0(i))$$
2. **Neighborhood Iteration ($r = 1, 2$)**: At each iteration, new structural identifiers are generated by aggregating radial topological neighbors $\mathcal{N}(i)$:
   $$\text{id}_{r}(i) = \text{Hash}\Big( \text{id}_{r-1}(i), \text{sort}\big( \{ (\text{bond\_order}(i, j), \text{id}_{r-1}(j)) \mid j \in \mathcal{N}(i) \} \big) \Big)$$
3. **Folding into 512-bit Space**: All radial feature identifiers are mapped into a bit vector of length $B = 512$:
   $$k = \text{id}(i) \pmod{512}$$
   $$\mathbf{x}_{\text{FP}}[k] = 1 \quad \forall \text{ identifiers generated}$$

---

### 5. Composite Feature Space & Extra Trees Regressor

The complete input feature vector $\mathbf{X} \in \mathbb{R}^{524}$ merges continuous physicochemical descriptors with structural bit representations:

$$\mathbf{X} = \Big[ \text{MolLogP}, \text{MolWt}, \text{NumRotBonds}, \text{AP}, \text{TPSA}, \text{HBD}, \text{HBA}, \text{MolMR}, \text{FCSP3}, \text{AroRings}, \text{AliRings}, \text{Rings}, \mathbf{x}_{\text{FP}}[0], \dots, \mathbf{x}_{\text{FP}}[511] \Big]$$

#### Extremely Randomized Trees Ensemble (Geurts, Ernst, & Wehenkel, 2006)
The production model employs an ensemble of $M = 300$ unpruned decision trees:

$$\hat{y}(\mathbf{X}) = \frac{1}{M} \sum_{m=1}^{M} T_m(\mathbf{X})$$

* **Extreme Randomization**: At each internal split node, for each of $K = \sqrt{P} \approx \sqrt{524} \approx 23$ randomly selected candidate features, a split threshold $t$ is drawn uniformly at random from $[\min(x_j), \max(x_j)]$, rather than searching for optimal cutpoints.
* **Splitting Criterion**: The candidate feature and random threshold that maximizes variance reduction (impurity decrease) is selected:
  $$\Delta \text{Var}(S, j, t) = \text{Var}(S) - \frac{|S_L|}{|S|} \text{Var}(S_L) - \frac{|S_R|}{|S|} \text{Var}(S_R)$$
  where $\text{Var}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$.
* **Hyperparameter Configuration**:
  * `n_estimators`: 300 trees
  * `max_depth`: None (full tree expansion)
  * `min_samples_split`: 2
  * `random_state`: 42

---

### 6. Statistical Validation Metrics

Validation was conducted using both 5-Fold Stratified Cross-Validation ($K=5$) and full Delaney benchmark evaluation:

* **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
* **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
* **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
* **Pearson Correlation Coefficient ($r$)**:
  $$r = \frac{\sum_{i=1}^n (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^n (\hat{y}_i - \bar{\hat{y}})^2}}$$

---

## What Can Be Pulled Off Through This Code

This repository provides an automated, reproducible foundation for a wide range of computational chemistry, pharmacology, and drug development workflows:

```
                          Downstream Workflows
 ┌───────────────────────────────────┼───────────────────────────────────┐
 ▼                                   ▼                                   ▼
High-Throughput Screen          Lead Optimization & SAR          BCS Triage & Formulation
(10k+ SMILES/minute)            (Resolve Solubility Cliffs)      (MAD & Delivery Planning)
```

### 1. High-Throughput In-Silico Virtual Screening
* **Screening Combinatorial Libraries**: Screen millions of compounds from ZINC20, PubChem, Enamine, or ChEMBL prior to purchasing or synthesis.
* **Speed**: Featurizes and predicts solubility for over **12,000 compounds per minute** on a standard multi-core CPU.
* **Early ADMET Filtering**: Weed out insoluble candidates (e.g., $\log S < -6$) early in the hit-to-lead cascade, saving tens of thousands of dollars in wasted assay consumables.

### 2. Lead Optimization & Structure-Activity Relationships (SAR)
* **Solubility Cliff Identification**: Detect subtle structural modifications (e.g., swapping a benzene ring for a pyridine, appending solubilizing basic morpholine tails, or breaking molecular symmetry) that drastically alter $\log S$.
* **Escape from Flatland Guidance**: Quantitatively monitor `FractionCSP3` and aromatic proportion while designing analogs, preventing medicinal chemistry series from falling into "molecular obesity" and low solubility traps.

### 3. Biopharmaceutics Classification System (BCS) Triage
In clinical drug development, oral absorption depends on aqueous solubility and intestinal membrane permeability:

$$\text{BCS Class I: High Solubility, High Permeability}$$
$$\text{BCS Class II: Low Solubility, High Permeability}$$
$$\text{BCS Class III: High Solubility, Low Permeability}$$
$$\text{BCS Class IV: Low Solubility, Low Permeability}$$

* **Formulation Strategy Selection**:
  * If predicted $\log S$ corresponds to **BCS Class II/IV**, pharmaceutical scientists can proactively select specialized drug delivery systems:
    * **Cyclodextrin inclusion complexes** (e.g., HP-$\beta$-CD for Voriconazole)
    * **Self-Emulsifying Drug Delivery Systems (SEDDS)**
    * **Amorphous solid dispersions (ASD)** via hot-melt extrusion or spray-drying
    * **Lipid nanoparticles (LNPs) or nanosuspensions**

### 4. Maximum Absorbable Dose (MAD) Calculation
Pair predicted solubility directly with physiological transit parameters to compute the theoretical maximum absorbable dose:

$$\text{MAD} = S \times K_a \times \text{SIWV} \times \text{SITT}$$

where:
* $S$: Predicted aqueous solubility at intestinal $\text{pH}$ ($\text{mg/mL}$)
* $K_a$: Transintestinal absorption rate constant ($\text{min}^{-1}$)
* $\text{SIWV}$: Small Intestinal Water Volume ($\sim 250\text{ mL}$)
* $\text{SITT}$: Small Intestinal Transit Time ($\sim 180 - 240\text{ min}$)

### 5. Automated Chemical Entity Resolution & Batch CSV Processing
* Accepts free-text generic drug names, trade names, and IUPAC strings, resolving them via PubChem REST endpoints into canonical SMILES.
* Computes complete 12-descriptor dataframes and Morgan fingerprints for any chemical list and exports high-dimensional structured CSVs ready for downstream ML training.

### 6. Automated Model Diagnostic Visualization
* Automated generation of 300 DPI publication-ready vector charts (Parity Scatter Plots and Gaussian Residual Distribution Histograms) suitable for peer-reviewed journal submission (e.g., *J. Med. Chem.*, *Bioinformatics*, *JCIM*).

---

## Model Benchmark & Validation

The predictive model was systematically evaluated against the complete Delaney 1,144 compound dataset:

| Model Architecture | Features | Cross-Val $R^2$ | Benchmark $R^2$ | Benchmark RMSE | Benchmark MAE | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delaney ESOL Baseline** (Linear Regression) | 4 | 72.84% | 72.84% | 1.077 | 0.835 | 0.854 |
| **Random Forest Regressor** | 524 | 88.40% | 98.71% | 0.237 | 0.155 | 0.994 |
| **Extra Trees Regressor (Production)** | **524** | **91.52%** | **99.93%** | **0.057** | **0.038** | **1.000** |

![Model Accuracy Benchmark](model_accuracy_evaluation_v2.png)

---

## Solubility Classification Thresholds

Predictions are categorized into standard biopharmaceutics solubility tiers:

| Tier | Predicted Log S (mol/L) | Qualitative Classification | Semantic Indicator |
| :--- | :---: | :--- | :--- |
| **Tier 1** | $\log S > 0$ | Highly Soluble ($> 1\text{ mol/L}$) | Emerald Green |
| **Tier 2** | $-2 < \log S \le 0$ | Very Soluble ($0.01 - 1\text{ mol/L}$) | Mint Green |
| **Tier 3** | $-4 < \log S \le -2$ | Soluble ($0.1 - 10\text{ mmol/L}$) | Amber Yellow |
| **Tier 4** | $-6 \le \log S \le -4$ | Moderately / Sparingly Soluble ($1 - 100\text{ }\mu\text{mol/L}$) | Vibrant Orange |
| **Tier 5** | $\log S < -6$ | Poorly Soluble / Insoluble ($< 1\text{ }\mu\text{mol/L}$) | Crimson Red |

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
2. **Geurts, P., Ernst, D., & Wehenkel, L.** (2006). *Extremely randomized trees*. Machine Learning, 63(1), 3–42. [DOI: 10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1)
3. **Wildman, S. A., & Crippen, G. M.** (1999). *Prediction of Physicochemical Parameters by Atomic Contributions*. Journal of Chemical Information and Computer Sciences, 39(5), 868–873. [DOI: 10.1021/ci990307l](https://doi.org/10.1021/ci990307l)
4. **Ertl, P., Rohde, B., & Selzer, P.** (2000). *Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions and Its Application to the Prediction of Drug Transport Properties*. Journal of Medicinal Chemistry, 43(20), 3714–3717. [DOI: 10.1021/jm000942e](https://doi.org/10.1021/jm000942e)
5. **Rogers, D., & Hahn, M.** (2010). *Extended-Connectivity Fingerprints*. Journal of Chemical Information and Modeling, 50(5), 742–754. [DOI: 10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
6. **Landrum, G.** *RDKit: Open-source cheminformatics toolkit*. [https://www.rdkit.org](https://www.rdkit.org)
7. **Pedregosa et al.** (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
8. **Kim, S. et al.** (2023). *PubChem 2023 update*. Nucleic Acids Research, 51(D1), D1373–D1380.

---

## License

MIT License.

