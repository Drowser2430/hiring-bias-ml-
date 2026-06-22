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
