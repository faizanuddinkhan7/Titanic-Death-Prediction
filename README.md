# Titanic Survival Prediction with PyTorch

An end-to-end Machine Learning pipeline built with **PyTorch**, **pandas**, **numpy**, and **matplotlib** to predict passenger survival on the Titanic with Stratified 5-Fold Cross-Validation.

---

## 📊 Cross-Validation Performance

- **Mean 5-Fold Accuracy:** **85.07% (± 1.37%)**
  - Fold 1: **86.59%** (AUC: 0.9172)
  - Fold 2: **84.83%** (AUC: 0.8867)
  - Fold 3: **82.58%** (AUC: 0.8604)
  - Fold 4: **85.39%** (AUC: 0.8683)
  - Fold 5: **85.96%** (AUC: 0.8791)
- **Overall Out-Of-Fold (OOF) Accuracy:** **85.07%** (758 / 891 correct)
- **Overall OOF ROC-AUC:** **0.8807**
- **Overall OOF Precision:** **0.8542**
- **Overall OOF Recall:** **0.7368**
- **Overall OOF F1-Score:** **0.7912**

---

## 🏗️ Architecture & Pipeline

1. **Exploratory Data Analysis (`src/eda.py`):**
   - Conducted with `pandas`, `numpy`, and `matplotlib`.
   - Identified critical survival factors (Sex: 74.2% female vs 18.9% male, Class: 63% 1st vs 24% 3rd, Title priority, and Family Size dynamics).
   - Generates and saves visual analysis to `plots/eda_summary.png`.

2. **Data Cleaning & Feature Engineering (`src/preprocess.py`):**
   - Missing `Age` imputed using median grouped by `(Pclass, Title)`.
   - Missing `Fare` in test set imputed using median grouped by `(Pclass, Embarked)`.
   - Missing `Embarked` imputed with mode (`'S'`).
   - Cabin transformed to extract deck letter (`A-G`, `'M'` for missing).
   - Features engineered: `Title`, `FamilySize`, `IsAlone`, `FamilyGroup` (Alone, Small, Large), `TicketFreq`, `IsChild`, `IsMother`.
   - Continuous features normalized via `StandardScaler`. Categorical features one-hot encoded.

3. **PyTorch Tabular Neural Network (`src/model.py`):**
   - Multi-layer perceptron with residual skip connections (`ResBlock`).
   - Batch normalization and dropout (0.25) to prevent overfitting.
   - SiLU non-linearities and Kaiming normal weight initialization.

4. **Training & Ensembling (`src/train.py`):**
   - 5-Fold Stratified Cross-Validation strictly fitting preprocessors per fold to prevent leakage.
   - `AdamW` optimizer + `CosineAnnealingLR` scheduler + `BCEWithLogitsLoss`.
   - Generates ensemble test predictions saved to `submission.csv`.
   - Plots training curves to `plots/training_curves.png`.

5. **Test Suite (`tests/test_pipeline.py`):**
   - 11 unit and integration tests verifying data integrity, model gradients, tensor dimensions, trained weights, and submission file specifications.

---

## 🚀 How to Run

### Run Exploratory Data Analysis
```bash
.venv/bin/python src/eda.py
```

### Train Model & Generate Submission
```bash
.venv/bin/python src/train.py
```

### Run Full Test Suite
```bash
.venv/bin/python tests/test_pipeline.py
```

