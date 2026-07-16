# CreditGuard — Loan Default Risk Prediction System

CreditGuard is an end-to-end machine learning system that predicts the probability that a loan applicant will default, based on their financial and credit history profile at the time of application. The project was built step-by-step — from raw data exploration, through preprocessing and model comparison, to a working REST API — with an emphasis on documenting *why* each decision was made, not just what was done.

---

## 1. Problem Definition

**Goal:** Given an applicant's financial profile (income, credit history, existing debt, employment, etc.), predict the **risk that they will default** on a loan.

Framed as **binary classification**:
- `0` = Non-default (loan expected to be repaid)
- `1` = Default (loan expected to default / become delinquent)

**Scope boundary (deliberately defined early on):** The model predicts a *risk score / probability*, not an approve/reject decision. Approving or rejecting a loan is a business decision that combines this risk score with business rules (risk appetite, pricing, regulation) — that layer is out of scope for the model itself.

---

## 2. Dataset

- **Source:** LendingClub loan data (`loans_full_schema.csv`) — 10,000 rows, 56 original columns.
- No ready-made binary "default" label existed in the raw data — it had to be derived (see below).

---

## 3. Target Variable: Deriving `default` from `loan_status`

The raw `loan_status` column had 6 categories, with the following distribution:

| Status | Count | % |
|---|---|---|
| Current | 9,375 | 93.75% |
| Fully Paid | 447 | 4.47% |
| In Grace Period | 67 | 0.67% |
| Late (31–120 days) | 66 | 0.66% |
| Late (16–30 days) | 38 | 0.38% |
| Charged Off | 7 | 0.07% |

### Decisions made and reasoning

- **Charged Off = 1 (default):** unambiguous — the lender has already written this off as unrecoverable.
- **Late (16–30 days) and Late (31–120 days) = 1 (default):** these loans have already missed scheduled payments. Rather than leaving them as a separate ambiguous "in progress" bucket, they were merged into the default class. This was also a practical necessity — "Charged Off" alone had only 7 rows, far too few to train on. Merging the late-payment categories brought the positive class up to a still-small-but-more-usable 111 rows.
- **Current, Fully Paid, In Grace Period = 0 (non-default):** "Current" loans are the largest category by far (93.75%). These loans haven't defaulted *yet*, but their final outcome is technically unknown — some fraction could still default in the future. Dropping all "Current" rows was considered (to keep the target perfectly clean) but rejected, since it would have shrunk the usable dataset from 10,000 rows to ~625 — too small to train a reliable model. This is a **deliberate, documented trade-off**: some label noise was accepted in exchange for a usable dataset size.

**Result:** `default` = 1 for 111 rows (1.11%), `default` = 0 for 9,889 rows (98.89%) — a **severely imbalanced target**, which shaped nearly every downstream decision (stratified splitting, `class_weight='balanced'`, threshold tuning).

---

## 4. Feature Selection & Cleaning

### 4.1 Leakage columns — dropped

`balance`, `paid_total`, `paid_principal`, `paid_interest`, `paid_late_fees` — these describe the loan's state *after* it has been disbursed and partially repaid. At the point a real prediction would be needed (loan approval time), none of these values exist yet. Including them would let the model "cheat" using information from the future.

### 4.2 Redundant pre-computed risk — dropped

`grade`, `sub_grade` — LendingClub's own internal risk grading. Keeping these would let the model simply learn to reproduce an existing risk score rather than learn from the underlying applicant features, defeating the purpose of building a model from raw data.

### 4.3 High-missing-value columns — dropped

`annual_income_joint`, `verification_income_joint`, `debt_to_income_joint` (only ~15% populated — joint-application only), `months_since_last_delinq`, `months_since_90d_late`, `months_since_last_credit_inquiry`.

### 4.4 Identifiers / low-value / high-cardinality — dropped

- `Unnamed: 0` — leftover pandas index, no information.
- `emp_title` — free-text job title, extremely high cardinality; would need NLP-level processing to be usable, out of scope.
- `state` — high cardinality (50 categories); dropped for simplicity rather than one-hot-encoding into 50 extra columns.
- `issue_month`, `initial_listing_status`, `disbursement_method` — not leakage (they are known at approval time), but judged unlikely to add meaningful predictive value at this stage, and dropped to keep the feature set focused.

### 4.5 Zero-variance column — dropped, verified numerically (not just visually)

`num_accounts_120d_past_due` had a KDE/box plot that looked "empty" — this was initially a point of confusion, since several *other* columns (`tax_liens`, `public_record_bankrupt`, `current_accounts_delinq`) also looked almost entirely flat/zero visually. Checking `.value_counts()` and `.nunique()` for all of them showed the difference: `num_accounts_120d_past_due` had **exactly one unique value (0.0) across every row** — genuinely zero variance, safe to drop. The other "mostly zero" columns had a small but non-zero fraction of meaningful values (e.g., `tax_liens` was ~97.5% zero, but the remaining ~2.5% is a strong, real default-risk signal — rare does not mean useless). This distinction — verifying with `nunique()`/`value_counts()` rather than dropping anything that merely *looks* sparse on a graph — was an explicit lesson learned during EDA.

### 4.6 Original superseded columns — dropped

`loan_status` (replaced by the derived `default` column — keeping the original would itself be leakage, since it's the direct source of the target) and `loan_purpose` (replaced by `loan_purpose_grouped`, below).

**Final feature set: 34 columns** (down from the original 56).

---

## 5. Exploratory Data Analysis

### 5.1 Numerical features

KDE plots and box plots were generated for all numerical columns. Several were confirmed to be strongly right-skewed with heavy outliers — including `annual_income`, `total_credit_limit`, `total_credit_utilized`, `total_collection_amount_ever`, and `debt_to_income` (which had a maximum value of 469 against a 75th percentile of only ~25).

**Decision: outliers were investigated, not removed.** The top debt-to-income and income/credit-limit values were checked individually (`.sort_values()`) and found to decrease smoothly rather than jump abruptly — indicating genuine (if extreme) data rather than data-entry errors. In a credit-risk context, these extreme cases are often exactly the examples a model most needs to learn from, so removing them would likely have discarded valuable signal.

### 5.2 Categorical features

Count plots were generated for `homeownership`, `verified_income`, `loan_purpose`, and `application_type`.

- `homeownership` (MORTGAGE / RENT / OWN) and `verified_income` (Verified / Source Verified / Not Verified) were both reasonably balanced across categories.
- `application_type` (individual / joint) was imbalanced (~85/15) but not extreme.
- `loan_purpose` was heavily skewed: `debt_consolidation` alone made up 51.4%, while several categories (`renewable_energy`, `vacation`, `moving`, etc.) each made up under 1%.

### 5.3 Rare-category grouping: `loan_purpose_grouped`

Because `loan_purpose` had 11 categories with several representing well under 1% of rows, one-hot encoding it directly would have produced multiple extremely sparse columns carrying almost no learnable signal, with a real risk of overfitting to noise in categories with only a handful of examples.

**Approach:** any category representing less than 5% of rows was grouped into a new `Misc` category (care was taken to name it distinctly from the existing `other` category already present in the data, to avoid a naming collision). This reduced the column to 5 clean categories: `debt_consolidation`, `credit_card`, `other`, `home_improvement`, `Misc`.

---

## 6. Train/Test Split

```python
X = data.drop(columns=['default'])
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**`stratify=y` was used deliberately** given the severe class imbalance (1.1% positive). Without it, a random split risks producing a train or test set with a meaningfully different default rate purely by chance, which would make evaluation unreliable. Stratification was verified after the split by comparing `y_train.value_counts(normalize=True)` and `y_test.value_counts(normalize=True)` — both matched the original ~98.9% / ~1.1% split.

---

## 7. Preprocessing Pipeline

Built with scikit-learn's `ColumnTransformer`, split into four column groups, each requiring a different treatment:

| Group | Columns | Steps | Reasoning |
|---|---|---|---|
| **1** | `emp_length`, `debt_to_income` | `SimpleImputer(strategy='median')` → `PowerTransformer` | The only two columns with missing values (652 and 16 missing in the training set respectively). Median was chosen over mean because both columns are skewed — mean would be distorted by outliers. `PowerTransformer` then corrects skew and standardizes scale. |
| **2** | All other numerical columns | `PowerTransformer` | No missing values, but several columns are still skewed and on very different scales (e.g., `annual_income` in the hundred-thousands vs. `delinq_2y` in single digits) — `PowerTransformer` handles both skew-correction and scaling in one step. |
| **3** | `verified_income` | `OrdinalEncoder(categories=[['Verified','Source Verified','Not Verified']])` | Has a genuine natural order (degree of income verification), so ordinal encoding is appropriate. The explicit `categories` order had to be specified, since scikit-learn would otherwise default to alphabetical order. |
| **4** | `homeownership`, `application_type`, `loan_purpose_grouped` | `OneHotEncoder(handle_unknown='ignore')` | No defensible natural order between categories. `homeownership` was specifically considered for ordinal encoding (e.g., OWN > MORTGAGE > RENT by financial stability) but rejected — the ranking isn't confidently defensible, and forcing an arbitrary numeric order risks encoding a false assumption into the model. `handle_unknown='ignore'` protects against categories at inference time that weren't seen during training. |

After transformation, the feature count grew from 34 to **42** — entirely due to one-hot encoding expanding `homeownership` (3), `application_type` (2), and `loan_purpose_grouped` (5) into 10 binary columns.

---

## 8. Model Training & Comparison

All models were trained with `class_weight='balanced'` to counteract the ~99:1 class imbalance, using the same preprocessing pipeline via a single combined `Pipeline([('preprocessor', ...), ('model', ...)])`.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **Logistic Regression** | 0.027 | **0.68** | **0.052** |
| Decision Tree | 0.042 | 0.045 | 0.043 |
| Random Forest | 0.000 | 0.000 | 0.000 |

### What was observed

- **Logistic Regression** achieved by far the best Recall (68%) — correctly identifying most actual defaulters in the test set, though at the cost of many false positives (low Precision).
- **Decision Tree** performed far worse on Recall (4.5%) despite also using `class_weight='balanced'`. With only ~89 positive examples in the training set, a single tree overfit to narrow, non-generalizing splits — a known failure mode of tree models on very small minority classes.
- **Random Forest** collapsed entirely to predicting the majority class for every row (Recall = Precision = 0). The ensemble's majority vote across many trees, each individually biased toward the dominant class, overwhelmed the `class_weight` adjustment.

**Conclusion:** the simplest model outperformed both tree-based alternatives on this dataset. This is a genuine and useful finding — model complexity is not a substitute for data volume, and on a small, heavily imbalanced dataset, a well-regularized linear model generalized noticeably better than either tree-based approach tried. **Logistic Regression was selected as the final model.**

An initial training run of Logistic Regression produced `ConvergenceWarning` and `RuntimeWarning: overflow encountered` messages. These were investigated rather than ignored: the transformed feature matrix (`preprocessor.fit_transform(X_train)`) was checked and found to have a normal, bounded range (min ≈ -5.76, max ≈ 89.4), confirming the final output was not corrupted — the overflow occurred transiently inside `PowerTransformer`'s internal fitting process on a few raw columns with extreme values (e.g., `debt_to_income` had a max of 469 against a 75th percentile of ~25, and `total_credit_limit`/`annual_income` had a small number of very high, but genuine, values). `max_iter` was increased to resolve the convergence warning; the overflow warning was determined to be non-fatal given the verified output range.

---

## 9. Threshold Tuning

The default classification threshold (0.5) was compared against lower thresholds using `predict_proba()`:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.2 | 0.015 | 0.91 | 0.029 |
| 0.3 | 0.017 | 0.82 | 0.034 |
| 0.4 | 0.022 | 0.77 | 0.044 |
| 0.5 | 0.027 | 0.68 | 0.052 |

**0.3 was selected as the operating threshold**, favoring Recall over Precision. Rationale: in a lending-risk context, failing to flag an actual defaulter (false negative) is typically more costly to a lender than mistakenly flagging a safe applicant as risky (false positive) — so a lower threshold that catches more true defaulters, even at the cost of more false alarms, better reflects the real-world cost asymmetry of this problem.

**Honest limitation:** Precision remains low (1–3%) across every threshold tested. This is primarily a consequence of the very small number of positive examples available (111 out of ~10,000 rows) rather than a flaw in any single modeling choice — no threshold adjustment can fully compensate for that little training signal. This is explicitly noted as the primary area for future improvement (more data, or resampling techniques like SMOTE).

---

## 10. Object-Oriented Design

The final model is wrapped in a small class hierarchy rather than left as loose notebook code, so that new model types can be added later without touching existing code:

```
src/
└── models/
    ├── base_model.py            # abstract base class
    └── credit_risk_model.py     # concrete implementation
```

**`base_model.py`** defines an abstract `BaseModel` class (via Python's `ABC`/`abstractmethod`) that specifies the contract every model in this project must follow: `train(X, y)`, `predict(X)`, `predict_proba(X)`, and `evaluate(X, y)`. No implementation lives here — it exists purely to enforce a consistent interface, mirroring an abstract class/interface pattern.

**`credit_risk_model.py`** defines `CreditRiskModel(BaseModel)`, the concrete implementation used in this project:
- The constructor takes the already-built `ColumnTransformer` preprocessor and a configurable `threshold` (default 0.3), and assembles them into a single scikit-learn `Pipeline` with `LogisticRegression(class_weight='balanced', max_iter=1000)`.
- `predict()` is implemented on top of `predict_proba()`, applying the class's own `threshold` attribute rather than scikit-learn's hardcoded 0.5 default — this is what allows the threshold-tuning decision from Section 9 to be baked into the reusable model object itself.
- `evaluate()` returns a dictionary of confusion matrix, precision, recall, and F1, so a full evaluation is a single method call.

This structure means adding a future model (e.g. a tuned Decision Tree or Random Forest once more data is available) only requires creating a new class that implements the same four methods — no changes to existing code are needed.

---

## 11. Model Persistence & Deployment (FastAPI)

The trained `CreditRiskModel` instance (preprocessing pipeline + classifier + threshold, all together) is serialized with `joblib`:

```python
import joblib
joblib.dump(model, 'src/models/trained_credit_risk_model.pkl')
```

This `.pkl` file is loaded once at API startup rather than retraining on every request.

### API (`src/api/main.py`)

A minimal FastAPI service exposes the trained model over HTTP:

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('src/models/trained_credit_risk_model.pkl')

@app.get("/")
def home():
    return {"message": "CreditGuard API is running"}

@app.post("/predict")
def predict(applicant: dict):
    df = pd.DataFrame([applicant])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return {
        "prediction": int(prediction[0]),
        "default_probability": float(probability[0])
    }
```

Run with:
```
uvicorn src.api.main:app --reload
```

The `/predict` endpoint accepts a JSON object describing a single applicant's raw (unprocessed) features, passes it straight through the same `Pipeline` used in training (so preprocessing — imputation, power-transform, scaling, encoding — happens identically in both training and inference), and returns both the binary prediction and the underlying default probability. This was tested via FastAPI's built-in interactive docs (`/docs`) using a hand-constructed sample applicant, and confirmed to return the same prediction and probability as the equivalent call made directly in the notebook — verifying the full pipeline behaves identically end-to-end.

---

## 12. Known Limitations & Future Work

- **Severe class imbalance** (1.1% positive class) is the single biggest constraint on model quality achieved so far. More labeled default examples, or resampling approaches like SMOTE, are the most likely way to meaningfully improve Precision.
- **Label noise** exists in the target because all "Current" loans were treated as non-default, even though a fraction may eventually default.
- Tree-based models (Decision Tree, Random Forest) underperformed here specifically because of the small positive class — this may not hold once more data or resampling is introduced, and is worth revisiting.
- Feature importance analysis was not the focus of this iteration (best suited to tree-based models once they perform reliably) but is a natural next step.
- The `dict`-based FastAPI request body is intentionally simple; a stricter Pydantic schema (explicit field names/types for all 34 features, with validation) would be a natural hardening step before any real deployment.
- Not yet implemented: Docker containerization and Kubernetes deployment, and CI-style automated tests for the pipeline and API.

---

## Tech Stack

- **Language:** Python
- **Data/ML:** pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, SimpleImputer, PowerTransformer, OneHotEncoder, OrdinalEncoder, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
- **Visualization:** seaborn, matplotlib
- **Serving:** FastAPI, uvicorn, joblib
- **Planned:** Docker / Kubernetes (deployment), automated tests

---

## Project Structure

```
CreditGuard/
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│   ├───processed
│   └───raw
│           loans_full_schema.csv
│
├───notebooks
│       01_eda.ipynb
│
├───src
│   ├───api
│   │       main.py
│   ├───data
│   ├───features
│   └───models
│           base_model.py
│           credit_risk_model.py
│           trained_credit_risk_model.pkl
│
└───tests
```