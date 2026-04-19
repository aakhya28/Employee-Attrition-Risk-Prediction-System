# Employee Attrition Risk Prediction System

A machine learning system to predict employee attrition risk using HR analytics data, deployed via an interactive Streamlit web application.

## Overview

Employee attrition is a critical challenge for organizations. This project builds a complete end-to-end ML pipeline that predicts which employees are at risk of leaving, helping HR teams take proactive retention decisions.

## Tech Stack

`Python` `Scikit-learn` `XGBoost` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Streamlit` `SMOTE` `Joblib`

## Project Structure

```
Employee-attrition-risk-prediction/
├── data/
│   └── Hr-employee-Attrition.csv
├── models/
│   ├── xgboost_attrition_model.joblib
│   ├── selected_features.joblib
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed 35 features from IBM HR Analytics dataset
- Removed 3 irrelevant features, retaining 32 for modeling
- Identified class imbalance: 83% No Attrition vs 17% Attrition

### 2. Feature Selection
- Used XGBoost feature importance scores to rank all features
- Applied threshold of 0.01 to eliminate low-importance features
- Reduced from 32 to 29 meaningful features

### 3. Model Training (4 Experiments)

| Block | Description |
|-------|-------------|
| Block 1 | Baseline — all 32 features |
| Block 2 | Feature selection — 29 features |
| Block 3 | SMOTE applied to handle class imbalance |
| Block 4 | Hyperparameter tuning with RandomizedSearchCV |

Models evaluated: Logistic Regression, Random Forest, XGBoost, SVM

### 4. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Used `class_weight='balanced'` for LR, RF, SVM
- Used `scale_pos_weight=5` for XGBoost

### 5. Hyperparameter Tuning
- RandomizedSearchCV with 20 iterations and 5-fold cross validation
- Optimized for F1-Score (appropriate for imbalanced data)

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.7823 | 0.3333 | 0.6410 | 0.4386 | 0.7836 |
| Random Forest | 0.8435 | 0.3871 | 0.3077 | 0.3429 | 0.7146 |
| **XGBoost** | **0.8707** | **0.5128** | **0.5128** | **0.5128** | **0.7934** |
| SVM | 0.7721 | 0.3250 | 0.6667 | 0.4370 | 0.7760 |

**Best Model: XGBoost (Tuned with SMOTE)**
- F1 improved from 0.18 (baseline) to 0.51 (tuned)
- Perfectly balanced Precision and Recall at 0.5128

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/aakhya28/Employee-Attrition-Risk-Prediction-System
cd Employee-attrition-risk-prediction
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model**
```bash
python train.py
```

**5. Run the Streamlit app**
```bash
streamlit run app.py
```

## Key Findings

- **OverTime, JobLevel, MaritalStatus** are the strongest predictors of attrition
- Employees with high overtime and low job level are at highest risk
- SMOTE significantly improved recall for the minority attrition class
- XGBoost outperformed all other models on F1 and ROC-AUC

## Dataset

IBM HR Analytics Employee Attrition Dataset — 1470 records, 35 features
