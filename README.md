# Customer Transaction Prediction (PRCP-1003)

**Project done by: Santoshkumar G**

Supervised binary classification on anonymized banking data: predict which customers will make a transaction in the future, regardless of the amount.

## Problem
- Domain: Banking / Finance
- Target: `target` (0 = no transaction, 1 = transaction)
- Features: 200 anonymized numeric variables (`var_0` to `var_199`) plus `ID_code`
- Challenge: imbalanced classes (about 90% vs 10%) and weak signal per feature

## Dataset
The dataset is not stored in this repository because of its size (GitHub rejects files over 100 MB).
1. Download: https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1003-CustTransPred.zip
2. Unzip and place the training file at `data/train.csv`

## Approach
1. Data quality checks and light EDA (80 missing values found, no duplicates)
2. Stratified 80/20 split, mean imputation and scaling fitted on training data only
3. Imbalance handling: class weights and SMOTE (training data only)
4. Model comparison: Logistic Regression, Random Forest, XGBoost, LightGBM
5. Hyperparameter tuning with RandomizedSearchCV and 5-fold Stratified K-Fold
6. Threshold selection from the precision-recall curve

## Results (validation set)
| Model | ROC-AUC | F1 |
|---|---|---|
| LightGBM | 0.875 | 0.462 |
| XGBoost | 0.864 | 0.459 |
| Logistic Regression (balanced) | 0.860 | 0.418 |
| Logistic Regression + SMOTE | 0.857 | 0.422 |
| Random Forest | 0.801 | 0.000 |

Tuned LightGBM: ROC-AUC about 0.888, recall 0.71 and precision 0.39 for the transacting class at the default 0.5 threshold.

**Recommended for production:** tuned LightGBM (best ROC-AUC, fast training, handles imbalance, easy to retrain). Logistic Regression is a simple, interpretable fallback.

## Run locally
```bash
git clone https://github.com/<your-username>/customer-transaction-prediction.git
cd customer-transaction-prediction
pip install -r requirements.txt
jupyter notebook PRCP-1003_Customer_Transaction_Prediction.ipynb
```

## Repository structure
```
.
├── PRCP-1003_Customer_Transaction_Prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── data/            (not tracked; add train.csv here)
```

## Tech stack
Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, matplotlib, seaborn

## Author
**Santoshkumar G**
