# Customer Churn Prediction with Explainable ML

An end-to-end machine learning project that predicts customer churn for a 
telecom company, going beyond a standard tutorial by comparing multiple 
models, explaining predictions with SHAP, and deploying via a containerized 
FastAPI service with a Streamlit frontend.

## Problem

Customer churn — when a subscriber leaves or downgrades — is costly for 
subscription-based businesses. This project predicts which customers are 
at risk of churning so retention efforts can be targeted proactively, 
using the Telco Customer Churn dataset (7,043 customers, 21 features).

## Approach

**1. Exploratory Data Analysis**
- Identified and fixed a hidden data quality issue: `TotalCharges` was 
  stored as text due to blank strings for new customers (tenure=0)
- Found the target is imbalanced (~73% No / ~27% Yes churn)
- Discovered `tenure` is bimodal — large clusters of both brand-new and 
  long-tenured customers
- Found strong multicollinearity between `TotalCharges`, `tenure`, and 
  `MonthlyCharges` (as expected, since TotalCharges ≈ tenure × MonthlyCharges) 
  — dropped `TotalCharges` before modeling

**2. Model Comparison**

Rather than deploying a single model, four were trained and compared, 
optimizing for **recall** on the churn class — since missing an actual 
churner is costlier than a false alarm for this business problem:

| Model | Accuracy | Recall (Churn) | Precision (Churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 74% | 76% | 51% | 0.836 |
| Random Forest | 78% | 50% | 61% | 0.815 |
| XGBoost (default) | 76% | 69% | 54% | 0.816 |
| **XGBoost (tuned)** | 74% | **80%** | 51% | **0.841** |

Random Forest had the highest raw accuracy but the worst recall — a 
reminder that accuracy is a misleading metric on imbalanced data. 
**Tuned XGBoost (via GridSearchCV, 5-fold CV)** was selected as the final 
model for its superior recall and ROC-AUC.

**3. Explainability (SHAP)**

SHAP values were used to explain what actually drives churn predictions:
- **Contract type** — month-to-month contracts strongly increase churn risk; 
  longer contracts strongly reduce it
- **Tenure** — new customers are at higher risk, consistent with EDA
- **Internet service (Fiber optic)** and **payment method (Electronic check)** 
  also correlate with higher churn

These findings translate directly into retention strategy: prioritize 
contract upgrades and early-tenure engagement for at-risk customers.

**4. Deployment**

- **FastAPI** backend serving predictions via a `/predict` endpoint, with 
  Pydantic schema validation
- **Streamlit** frontend for an interactive, non-technical UI
- **Docker** containerization for reproducible deployment

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Docker

## Project Structure
├── notebook/                  # Full EDA + modeling notebook
├── data/                       # Dataset
├── app/
│   ├── main.py                 # FastAPI backend
│   └── streamlit_app.py        # Streamlit frontend
├── churn_model.pkl             # Trained tuned XGBoost model
├── scaler.pkl
├── model_columns.json
├── requirements.txt
├── Dockerfile
└── README.md

## How to Run

**Option 1 — Docker (recommended)**
```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```
API available at `http://localhost:8000/docs`

**Option 2 — Local**
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
streamlit run app/streamlit_app.py
```

## Key Takeaways

- Accuracy alone is misleading on imbalanced classification problems — 
  recall and ROC-AUC were prioritized instead
- Hyperparameter tuning (GridSearchCV) improved XGBoost's recall from 
  69% to 80%
- SHAP explainability turns a black-box prediction into an actionable 
  business insight

