# Motor Condition Monitoring & Regression Models

This project implements a complete pipeline for **motor condition monitoring** and **temperature/voltage prediction** using several regression and classification models.  
It includes data preprocessing, feature engineering, model training with hyperparameter tuning, and anomaly detection.

This project was developed as part of an **8-week course**. Some of the code and utility functions were provided by the course instructors. The main development and implementation of the solution were carried out collaboratively by Lucas Tramonte and myself.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models](#models)
- [Results & Export](#results--export)

---

## Overview
The goal of this project is to:
- Clean and smooth raw signals from multiple motors
- Preprocess and make some adjusments on the data
- Build time-based features using sliding windows
- Train and compare multiple regression and classsification models to perform several tasks
- Detect abnormal motor behavior using a regression-based fault detector
- Export predictions for submission

---

## Features
**Type of features** : State of all the motors that composes the arm robot
**Data preprocessing**: outlier removal, low-pass filtering, interpolation for missing data, normalization 
**Feature engineering**: sliding windows to capture temporal dynamics  

---

## Models Used

Regression models :
* **Linear Regression**
* **Ridge Regression**
* **Lasso Regression**
* **ElasticNet Regression**
* **Decision Tree Regression**

Classification models :
* **RandomForest Classifier**
* **Logistic Regression**
* **Decision Tree Classifier**
* **Gradient Boosting**
* **Support Vectir Machine**

Model evaluation is done via **Accuracy** and **F1-score** for classification and **RMSE** for regression.

---

## Results & Export

Predictions for each motor are consolidated into a single DataFrame and exported as CSV for submission (and teacher evaluation).

Some intermediate results of our models are available in the notebooks (WP_1, WP_2 and WP_3). Final submission was made through **main_data_challenge.ipynb**.