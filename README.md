# Hiring Bias Detection on Amazon SageMaker

An end-to-end MLOps pipeline for detecting, gating, and monitoring algorithmic bias in hiring-style classifiers, built on Amazon SageMaker using the UCI Adult Income dataset as a proxy for a candidate screening decision.

**Course:** AAI-540-02 Machine Learning Operations<br>
**Institution:** Shiley-Marcos School of Engineering, University of San Diego<br>
**Instructor:** Professor Rod Albuyeh<br>
**Authors:** Darius Rowser, Mustafa Yunus<br>
**Date:** June 2026

---

## Project Overview

This project demonstrates that algorithmic fairness can be integrated as a first-class concern at every stage of an MLOps lifecycle. The work covers data preparation and feature engineering, CI/CD pipeline orchestration with an automated fairness gate, post-training bias auditing, and production monitoring of a deployed endpoint.

**Headline result:** All three trained classifiers (Logistic Regression, Random Forest, XGBoost) achieve accuracy above 84%, but all three fail the EEOC four-fifths rule with Disparate Impact values between 0.29 and 0.53 across sex, race, and age. The automated CI/CD gate correctly blocks model registration when fairness thresholds are violated. The production monitoring stack provides ongoing visibility into data quality drift, model quality drift, and fairness drift.

---

## Repository Structure

```
hiring-bias-ml/
├── README.md
├── notebooks/
│   ├── 01_data_prep_and_feature_store.ipynb     Data ingestion, EDA, splits, Feature Store
│   ├── 02_cicd_pipeline.ipynb                   SageMaker Pipelines CI/CD with fairness gate
│   ├── 03_monitoring_FIXED.ipynb                Endpoint deploy + Model Monitor + Clarify
│   └── 04_bias_audit_colab.ipynb                Standalone bias audit (runs on Google Colab)
├── paper/
│   └── Rowser_Yunus_Bias_Hiring_AAI540.docx     Final APA-formatted project paper
├── images/
│   ├── pretraining_bias.png                     Label distribution by protected group
│   ├── disparate_impact.png                     DI across three models and three attributes
│   └── shap_summary.png                         SHAP feature importance for XGBoost
└── .gitignore
```

---

## Data

**Source:** UCI Machine Learning Repository, Adult Income dataset (Becker & Kohavi, 1996)<br>
**URL:** https://archive.ics.uci.edu/ml/datasets/adult<br>
**Records:** 48,842 raw, 45,222 after dropping null values<br>
**Features:** 14 (6 numeric, 8 categorical) plus a binary income label<br>
**Favorable class:** Income greater than $50,000 per year (24.8% of cleaned dataset)

The raw data is pulled directly from the UCI repository at notebook runtime and is not stored in this repository. Within AWS, all processed data and model artifacts live in S3 under:

```
s3://hiring-bias-adult-income-mustafa/hiring-bias-project/
├── data/
│   ├── raw/              Raw Adult Income CSVs as downloaded from UCI
│   ├── train/            Training split (70%)
│   ├── validation/       Validation split (15%)
│   ├── test/             Test split (15%)
│   └── pipeline-input/   Consolidated training data for SageMaker Pipelines
├── models/
│   └── local-train/      Trained model artifacts (XGBoost) packaged as model.tar.gz
├── baseline/             Model Monitor baseline statistics and constraints
├── monitoring/           Hourly monitoring schedule outputs
├── datacapture/          Real-time endpoint request/response captures
├── ground-truth/         JSON-lines ground truth files for model quality monitoring
└── clarify/              SageMaker Clarify bias and explainability reports
```

A SageMaker Feature Store Feature Group named `hiring-bias-feature-group` provides both offline (S3) and online (low-latency) access to engineered features. A 5,000-record sample from the training partition was ingested during initial development to validate the schema and ingestion pipeline.

---

## How to Run

**Prerequisites:**
- AWS account with SageMaker, S3, and CloudWatch permissions
- Python 3.10 or higher
- SageMaker SDK v2.x (`pip install "sagemaker<3.0"`)
- An execution role with full SageMaker access

**Run order:**

1. **`01_data_prep_and_feature_store.ipynb`** — Creates the S3 bucket, downloads the Adult Income data, performs EDA, generates train/val/test splits, and creates the SageMaker Feature Group.

2. **`02_cicd_pipeline.ipynb`** — Defines and executes the SageMaker Pipeline with the following steps: Process → Train → Evaluate → ConditionStep (accuracy + Disparate Impact gate) → either CreateModel + RegisterModel + Transform OR FailStep.

3. **`03_monitoring_FIXED.ipynb`** — Deploys the registered model behind a real-time SageMaker endpoint with 100% data capture, generates 300 test invocations, configures Model Monitor data quality and model quality schedules, runs SageMaker Clarify bias and explainability jobs, and creates the CloudWatch dashboard.

4. **`04_bias_audit_colab.ipynb`** *(optional, runs on Google Colab)* — Standalone bias audit that trains Logistic Regression, Random Forest, and XGBoost locally and computes Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference, and SHAP explainability. Use this to reproduce the headline fairness numbers without AWS costs.

**Estimated runtime:**
- Notebook 01: 10-15 minutes
- Notebook 02: 25-35 minutes
- Notebook 03: 45-60 minutes (endpoint deploy + baseline jobs + Clarify jobs)
- Notebook 04: 4-6 minutes on Colab free tier

---

## Key Results

| Model | Accuracy | AUC | DI (sex) | DI (race) | DI (age) | Passes Gate |
|-------|----------|-----|----------|-----------|----------|-------------|
| Logistic Regression | 84.50% | 0.902 | 0.287 | 0.526 | 0.332 | No |
| Random Forest | 85.68% | 0.914 | 0.309 | 0.513 | 0.333 | No |
| XGBoost (pipeline) | 87.02% | 0.926 | 0.320 | 0.524 | 0.288 | No |

All three models fail the EEOC four-fifths rule on all three protected attributes. The CI/CD gate is configured to block model registration in this state.

**SHAP top features (XGBoost):**

1. `marital_status_Married-civ-spouse` (mean abs SHAP: 1.116)
2. `age` (0.647)
3. `capital_gain` (0.543)
4. `education_num` (0.489)
5. `hours_per_week` (0.315)
6. `capital_loss` (0.152)
7. `sex_Female` (0.146)

Marital status acts as a proxy for sex, demonstrating that simply removing the sex feature would not eliminate the disparate impact.

---

## Deliverables

- **Project paper** — `paper/Rowser_Yunus_Bias_Hiring_AAI540.docx` (APA 7 format)
- **Three SageMaker notebooks** — 01 data prep, 02 CI/CD pipeline, 03 monitoring
- **Supplemental bias audit notebook** — 04 (runs on Colab)
- **Video demo** — submitted separately via Canvas

---

## References

Full reference list available in the paper. Key sources:

- Becker, B., & Kohavi, R. (1996). Adult dataset. UCI Machine Learning Repository.
- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness.
- EEOC. (1978). Uniform guidelines on employee selection procedures, 29 C.F.R. § 1607.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning.
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems.

---

## License

Academic project. Code is provided as-is for educational purposes.
