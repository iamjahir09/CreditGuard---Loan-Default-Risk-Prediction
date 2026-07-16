# 🏦 CreditGuard – Loan Default Risk Prediction System

CreditGuard is an end-to-end Machine Learning project that predicts whether a loan applicant is likely to default on a loan based on their financial profile and credit history.

Instead of making the loan approval decision, the system estimates the applicant's default risk and assigns a risk category (Low, Medium, or High), allowing financial institutions to make informed lending decisions.

---

## Features

- Loan Default Prediction using Machine Learning
- Applicant Risk Classification (Low / Medium / High)
- Data Preprocessing Pipeline
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Logistic Regression Model
- REST API using FastAPI
- Object-Oriented Design
- Modular Project Structure

---

## Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### API
- FastAPI
- Uvicorn

### Model Persistence
- Joblib

---

## Problem Statement

Financial institutions receive thousands of loan applications every day.

Evaluating every applicant manually is expensive and time-consuming.

CreditGuard predicts the probability that an applicant may default on a loan using historical lending data, helping banks identify high-risk applicants before approving loans.

---

## Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Data Preprocessing
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Threshold Optimization
      │
      ▼
FastAPI Deployment
```

---

## Machine Learning Pipeline

- Data Cleaning
- Missing Value Handling
- Feature Selection
- Feature Engineering
- Encoding
- Power Transformation
- Model Training
- Model Evaluation
- Probability Prediction
- Risk Classification

---

## Project Structure

```
CreditGuard/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── api/
│   └── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model

The final model uses **Logistic Regression** because it achieved the best balance between recall and generalization on this highly imbalanced dataset.

The complete preprocessing pipeline is integrated using Scikit-learn Pipelines, ensuring identical transformations during both training and inference.

---

## Future Improvements

- XGBoost
- LightGBM
- SMOTE for class imbalance
- Docker
- Kubernetes
- CI/CD Pipeline
- Unit Testing
- Model Monitoring

---

## Author

**Shaikh Jahir Atik**

B.E. Computer Science & Engineering (AI & ML)

GitHub: https://github.com/iamjahir09