# Hiring Bias Detection on Amazon SageMaker

An end-to-end MLOps pipeline for detecting, gating, and monitoring algorithmic bias in hiring-style classifiers, built on Amazon SageMaker using the UCI Adult Income dataset as a proxy for a candidate screening decision.

**Course:** AAI-540-02 Machine Learning Operations
**Institution:** Shiley-Marcos School of Engineering, University of San Diego
**Instructor:** Professor Rod Albuyeh
**Authors:** Darius Rowser, Mustafa Yunus
**Date:** June 2026

---

## Project Overview

This project demonstrates that algorithmic fairness can be integrated as a first-class concern at every stage of an MLOps lifecycle. The work covers data preparation and feature engineering, CI/CD pipeline orchestration with an automated fairness gate, post-training bias auditing, and production monitoring of a deployed endpoint.

**Headline result:** All three trained classifiers (Logistic Regression, Random Forest, XGBoost) achieve accuracy above 84%, but all three fail the EEOC four-fifths rule with Disparate Impact values between 0.29 and 0.53 across sex, race, and age. The automated CI/CD gate correctly blocks model registration when fairness thresholds are violated. The production monitoring stack provides ongoing visibility into data quality drift, model quality drift, and fairness drift.

---

## Repository Structure


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
│   ├── disparate_impact.png                     DI across three models and

---

## Data

**Source:** UCI Machine Learning Repository, Adult Income dataset (Becker & Kohavi, 1996)<br>
**URL:** https://archive.ics.uci.edu/ml/datasets/adult<br>
**Records:** 48,842 raw, 45,222 after dropping null values<br>
**Features:** 14 (6 numeric, 8 categorical) plus a binary income label<br>
**Favorable class:** Income greater than $50,000 per year (24.8% of cleaned dataset)

The raw data is pulled directly from the UCI repository at notebook runtime and is not stored in this repository. Within AWS, all processed data and model artifacts live in S3 under:
