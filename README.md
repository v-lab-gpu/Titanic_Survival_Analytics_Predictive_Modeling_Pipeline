# 🚢 Titanic Survival Analytics & Predictive Modeling Pipeline

An end-to-end machine learning pipeline built on the classic Titanic dataset — covering data cleaning, exploratory data analysis, multivariate storytelling, classification, class-imbalance handling, hyperparameter tuning, regression, and model persistence, all wrapped in a single reproducible notebook.

This project was built to demonstrate production-style ML workflow habits: leakage-safe preprocessing, justified data-cleaning decisions, multi-model benchmarking, and a saved, reloadable inference pipeline, not just a notebook of disconnected plots.

---

## 📌 Project Highlights

- **Leakage-free preprocessing** : all imputation, scaling, and encoding are fit exclusively on training data inside `sklearn` Pipelines, never on the full dataset.
- **Justified missing-data strategy** : every column's missing-value treatment (drop rows / impute / drop column) is decided programmatically based on missingness thresholds (< 5%, 5–30%, > 30%).
- **Three benchmarked classifiers**  Logistic Regression, Decision Tree, and Random Forest compared on Accuracy, Precision, Recall, F1, and ROC-AUC.
- **Class imbalance handling** : baseline vs. `class_weight="balanced"` vs. SMOTE oversampling, compared fairly (SMOTE applied only to training folds).
- **Hyperparameter tuning** : `GridSearchCV` (5-fold CV, F1-optimized) over Random Forest depth, estimators, and feature sampling, with out-of-bag (OOB) score reported.
- **Regression side-task** : a multivariate Linear Regression model predicting `fare`, evaluated with MAE, RMSE, R², and Adjusted R², including residual diagnostics for heteroscedasticity.
- **Model persistence** : the best-performing complete pipeline (preprocessing + estimator) is serialized with `joblib` and verified to reproduce identical predictions on raw, unprocessed input after reloading.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Model Persistence | joblib |

---

## 📂 Project Structure

```
├── titanic_analysis.ipynb          # Main pipeline notebook
├── titanic.csv                     # Raw dataset (seaborn's built-in Titanic data)
├── best_titanic_pipeline.joblib    # Saved, reloadable end-to-end model pipeline
├── charts/                         # All generated visualizations
│   ├── correlation_heatmap.png
│   ├── survival_sex_class.png
│   ├── survival_age.png
│   ├── fare_survival.png
│   ├── family_survival.png
│   ├── decision_tree.png
│   ├── *_confusion_matrix.png
│   ├── roc_curves.png
│   └── regression_residuals.png
└── outputs/                        # Metric tables (CSV)
    ├── model_comparison.csv
    ├── imbalance_comparison.csv
    └── regression_metrics.csv
```

---

## 🔍 Workflow Overview

### 1. Data Loading & Inspection
Loads the Titanic dataset via `seaborn.load_dataset`, exports a raw CSV snapshot, and inspects shape, dtypes, and descriptive statistics.

### 2. Missing Value Handling
A rules-based strategy determines the treatment for every column:
- **< 5% missing** → drop affected rows (`embarked`, `embark_town`)
- **5–30% missing** → median imputation (`age`)
- **> 30% missing** → drop column entirely (`deck`)

### 3. Bivariate & Correlation Analysis
Survival rates are broken down by sex, passenger class, and their combination using boolean masking, alongside a 6×6 correlation matrix (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`) visualized as a heatmap.

### 4. Multivariate Data Story
Four narrative-driven charts explore survival through the combined lens of sex, class, age, fare, and family size  each with a written interpretation.

### 5. Train/Test Split & Preprocessing
A stratified 80/20 split preserves class proportions. Numeric features are median-imputed and scaled; categorical features are mode-imputed and one-hot encoded all inside a `ColumnTransformer` fit only on training data.

### 6. Classification Modeling
Three classifiers (Logistic Regression, Decision Tree, Random Forest) are trained inside identical preprocessing pipelines and evaluated on held-out test data with confusion matrices and ROC curves.

### 7. Class Imbalance Handling
Compares three strategies baseline, class-weight balancing, and SMOTE oversampling — on Precision, Recall, and F1 to identify the best trade-off for the minority class.

### 8. Hyperparameter Tuning
`GridSearchCV` tunes the Random Forest across estimators, depth, and feature sampling strategy using 5-fold cross-validation optimized for F1, with OOB score reported as an additional validation signal.

### 9. Regression Side-Task
A separate Linear Regression model predicts `fare` from passenger attributes, evaluated with MAE, RMSE, R², and Adjusted R², plus a residual plot to check for heteroscedasticity.

### 10. Final Comparison & Model Persistence
All classification and regression metrics are consolidated into a single comparison table. The best-performing pipeline is saved with `joblib` and reloaded to confirm it reproduces identical predictions directly from raw input proving the artifact is deployment-ready.

---

## 📊 Example Results

> Exact values are generated at runtime and saved to `outputs/model_comparison.csv`. Typical performance on this dataset looks like:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.77 | ~0.72 | ~0.74 | ~0.85 |
| Decision Tree | ~0.79 | ~0.76 | ~0.70 | ~0.73 | ~0.82 |
| Random Forest (tuned) | ~0.82 | ~0.79 | ~0.75 | ~0.77 | ~0.87 |

Key takeaway: **sex, passenger class, and fare** are the strongest predictors of survival, while age alone shows limited discriminative power without multivariate context.

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

# 3. Run the notebook
jupyter notebook titanic_analysis.ipynb
```

The notebook is self-contained: it creates its own `charts/` and `outputs/` directories, downloads the dataset, and runs the full pipeline end to end.

---

## 🧠 Skills Demonstrated

- Data cleaning with quantitatively justified decisions
- Leakage-safe pipeline design with `sklearn.pipeline.Pipeline` and `ColumnTransformer`
- Handling imbalanced classification targets (class weighting, SMOTE)
- Model selection via cross-validated hyperparameter search
- Both classification and regression modeling in one workflow
- Model serialization and integrity verification for deployment readiness
- Clear, business-readable data storytelling through visualization

---

## 📄 License

This project is available under the MIT License, feel free to fork, adapt, and build on it.

---

## 👤 Author

Built as a portfolio project to showcase applied machine learning and data science engineering practices. Feel free to reach out via GitHub or LinkedIn for questions or collaboration.
