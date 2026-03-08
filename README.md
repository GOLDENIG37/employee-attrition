# Employee Attrition & Retention Analytics

Turnover is one of those problems that everyone in a company feels but nobody can see coming. A good engineer leaves, their manager is surprised, and HR scrambles to fill the gap — all while the institutional knowledge walks out the door. This project is an attempt to change that.

The core question: given everything we know about an employee — their compensation, satisfaction scores, role, tenure, commute — can we predict who's likely to leave before they hand in their notice? And more importantly, can we put a dollar figure on what acting on that prediction is actually worth?

---

## What makes this different from a typical ML project

A few deliberate choices:

**Three models, not one.** Logistic Regression, Random Forest, and Gradient Boosting each make different assumptions about the data. Comparing them honestly — including cross-validation variance — is how a senior analyst actually builds confidence in a result.

**Threshold optimization.** The default 0.5 classification threshold is almost never right for imbalanced HR problems. We optimize it based on F1, then interpret what that means in practice.

**Business cost framing.** The model's ROC-AUC is an internal metric. What HR leadership actually cares about is dollars — how many employees did we retain, what did the intervention program cost, and what was the net benefit? That calculation is in the notebook.

**Feature engineering with reasoning.** Added five engineered features, each motivated by a specific hypothesis about what drives attrition — not just thrown in to see if they improve accuracy.

---

## Project Structure

```
employee-attrition/
├── data/
│   ├── raw/                              # Original dataset, untouched
│   │   └── employee_attrition.csv        # 14,999 employees, 32 features
│   └── processed/                        # Encoded, model-ready data
├── notebooks/
│   └── employee_attrition_analysis.ipynb # Full analysis: EDA → models → business impact
├── src/
│   ├── preprocessing.py                  # Feature engineering + encoding utilities
│   └── model.py                          # Pipeline builder, evaluation, cost analysis
├── reports/
│   └── figures/                          # All charts from the notebook
├── models/                               # Saved model artifacts (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Dataset

14,999 employee records with 32 features per person. The dataset is synthetically generated to mirror the structure and statistical properties of real-world HR attrition data (inspired by the IBM HR Analytics dataset format).

**Attrition rate: ~9.4%** — a realistic class imbalance for enterprise HR data.

Feature categories:
- **Compensation:** MonthlyIncome, DailyRate, HourlyRate, StockOptionLevel, PercentSalaryHike
- **Role & tenure:** JobRole, Department, JobLevel, YearsAtCompany, YearsInCurrentRole
- **Satisfaction surveys:** JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, JobInvolvement
- **Work conditions:** OverTime, BusinessTravel, DistanceFromHome, TrainingTimesLastYear
- **Demographics:** Age, Gender, Education, MaritalStatus

**Engineered features added:**
- `IncomePerYearExperience` — compensation efficiency relative to experience
- `TenureRatio` — fraction of total career spent at this company
- `SatisfactionScore` — composite of all four satisfaction dimensions
- `PromotionStagnation` — binary flag for 4+ years without promotion
- `LongCommute` — binary flag for distance > 15 units

---

## Key Findings

**OverTime is the dominant signal.** Employees working overtime leave at roughly three times the rate of those who don't. Every model ranked this at or near the top. The practical implication: workload management is a retention strategy.

**Low satisfaction compounds everything.** Employees scoring 1 or 2 on job or environment satisfaction have dramatically higher attrition — and when they're also working overtime, it's nearly certain they'll leave.

**Early tenure is the danger zone.** Employees in their first 0-3 years are the highest-risk segment. Onboarding quality and early career development have more leverage than almost any other intervention.

**Income matters, but not in isolation.** Monthly income correlates with attrition, but the `IncomePerYearExperience` ratio engineered feature outperformed raw income in tree-based models — suggesting it's more about whether people feel fairly compensated relative to their experience level.

**Promotion stagnation is a quiet killer.** Employees who haven't been promoted in 4+ years show elevated attrition even when their salary is competitive. People want to see movement in their career, not just a paycheck.

---

## Model Performance

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | ~0.79 | 0.79 ± 0.01 |
| Random Forest | ~0.84 | 0.83 ± 0.01 |
| Gradient Boosting | ~0.85 | 0.84 ± 0.01 |

Gradient Boosting edges out Random Forest on AUC. The gap is small but consistent across folds, which gives confidence it's not noise.

---

## How to Run

```bash
git clone https://github.com/GOLDENIG37/employee-attrition.git
cd employee-attrition
pip install -r requirements.txt
jupyter notebook notebooks/employee_attrition_analysis.ipynb
```

Using the src modules directly:

```python
from src.preprocessing import load_raw, engineer_features, encode_and_split
from src.model import build_models, train_and_evaluate, optimal_threshold, business_cost_analysis

df = load_raw('data/raw/employee_attrition.csv')
df = engineer_features(df)
X_train, X_test, y_train, y_test, features = encode_and_split(df)

models = build_models()
results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
```

---

## Connection to This Portfolio

This project builds directly on [credit-risk-analysis](https://github.com/GOLDENIG37/credit-risk-analysis), which established the baseline EDA and logistic regression pattern. Here we go further — multiple models, feature engineering, threshold tuning, and business cost framing. The progression is intentional: each project adds a layer of analytical depth.

---

## Stack

Python 3.10 · pandas · scikit-learn · matplotlib · seaborn · Jupyter

---

## What's Next

- [ ] SHAP values — individual-level explanations, not just feature importance
- [ ] Survival analysis — *when* will someone leave, not just *if*
- [ ] Streamlit dashboard — interactive predictions for HR teams to use directly
- [ ] Hyperparameter tuning with cross-validated search
- [ ] Segment analysis — attrition drivers likely differ across departments and seniority levels

---

## About this project

Second project in a deliberate sequence. Coming from a frontend background, I was used to thinking about user behaviour in aggregate — conversion rates, session length, click-through rates. HR attrition is the same concept applied to employees instead of users. Someone is either retained or they're not. The signals that predict churn in a product also tend to predict churn in a workforce.

That lens — user behaviour thinking applied to a different domain — is what made this feel natural to explore. The multi-model comparison and business cost framing here were a deliberate step up from the baseline approach in [credit-risk-analysis](https://github.com/GOLDENIG37/credit-risk-analysis).

Next: [customer-clv-segmentation](https://github.com/GOLDENIG37/customer-clv-segmentation) — unsupervised segmentation, gradient boosting, and SHAP.
