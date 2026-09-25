# 🤖 Customer Churn Prediction

A Streamlit machine-learning application that estimates customer churn risk from demographics, services, contract details, and billing information.

## Overview

The project implements the complete workflow:

`Dataset → Data cleaning → Feature engineering → Classification → Evaluation → Prediction → Streamlit`

The application includes:

- Customer-level churn prediction with probability and risk bands
- Logistic Regression and Random Forest model comparison
- Interactive retention dashboard with filters and revenue-at-risk KPIs
- EDA charts, ROC curve, precision-recall curve, confusion matrix, and feature importance
- Downloadable customer summaries and CSV dashboard data

## Dataset

The source dataset contains 7,043 customer records. Eleven records have blank `TotalCharges` values and are removed for model training, leaving 7,032 records:

- 5,163 retained customers
- 1,869 churned customers
- 26.58% churn rate

## Model performance

Models are trained with a stratified 80/20 split, preprocessing, and five-fold cross-validation for Logistic Regression tuning. The final model is selected using F1 score.

| Model | Accuracy | Precision | Recall | F1 score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72.57% | 49.01% | 79.68% | 60.69% | 83.51% |
| Random Forest | 77.54% | 57.04% | 62.83% | 59.80% | 82.50% |

The current run selects Logistic Regression. The detailed metrics and confusion matrix are stored in `models/model_metrics.json`.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train or refresh the model artifacts:

```bash
python train_model.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Project structure

```text
customer-churn/
├── data/
│   └── customer_churn.csv
├── models/
│   ├── churn_model.pkl
│   ├── preprocessor.pkl
│   ├── model_metrics.json
│   └── evaluation charts
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
└── .gitignore
```

## Notes

`TotalCharges` is converted to numeric during training. The prediction is an estimate based on historical behavior, not a guarantee of future customer behavior.
