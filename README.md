# Breast Cancer Classification using Machine Learning

## A. Problem Statement

The objective of this project is to develop and compare different machine learning classification models for predicting whether a breast tumor is **benign or malignant**.

Five machine learning algorithms were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

The performance of these models was compared using the following evaluation metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

A Streamlit application was also developed to provide an interactive interface where a user can upload test data, select a model, and view its evaluation results.

---

## B. Dataset Description

The **Breast Cancer Wisconsin Diagnostic dataset** from the UCI Machine Learning Repository was used for this project.

The dataset contains:

- **569 observations**
- **30 numerical predictive features**
- **1 binary target variable (`diagnosis`)**

The target variable indicates whether the tumor is benign or malignant:

| Value | Diagnosis |
|---|---|
| 0 | Benign |
| 1 | Malignant |

The dataset was divided into training and testing sets using an **80:20 stratified split**.

| Dataset | Number of Samples |
|---|---:|
| Training Set | 455 |
| Testing Set | 114 |
| Total | 569 |

The original `id` column was removed before modelling because it is an identifier and does not provide useful predictive information. The dataset contained no missing values.

### Preprocessing and Modelling Approach

The dataset was first examined to understand its structure and check for missing values. The `id` column was removed before the models were trained.

The `diagnosis` values were converted into binary numerical labels, where `0` represents a benign tumor and `1` represents a malignant tumor.

The data was then divided into training and testing sets using stratified sampling. This helped maintain a similar proportion of benign and malignant cases in both sets.

Feature scaling using `StandardScaler` was applied to Logistic Regression and KNN. Decision Tree, Gaussian Naive Bayes and Random Forest were trained using the original unscaled feature values.

All five models were trained using the training data and evaluated on the same held-out test dataset.

---

## C. GitHub Repository

The complete project, including the notebook, source code, test data, trained models and supporting files, is available on GitHub.

**GitHub Repository:**  
https://github.com/SargamKamra04207/ml-assignment2-cancer-classification

---

## D. Models Used

The following five classification models were implemented:

| ML Model | Type |
|---|---|
| Logistic Regression | Linear Classification |
| Decision Tree | Tree-Based Classification |
| KNN | Distance-Based Classification |
| Gaussian Naive Bayes | Probabilistic Classification |
| Random Forest | Ensemble Classification |

### Model Comparison

All five models were evaluated on the same test dataset using the six performance metrics.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9649 | 0.9940 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

---

## E. Live Streamlit Application

The trained models were deployed using Streamlit Community Cloud.

**Live Streamlit Application:**  
https://sargamkamra04207-ml-assignment2-cancer-classificatio-app-rhenqc.streamlit.app/

The application allows the user to upload the test dataset, select any of the five trained models and view the corresponding evaluation results.

---

## F. Model-wise Observations

### Logistic Regression

Logistic Regression gave the strongest overall performance in this experiment. It achieved **96.49% accuracy**, the highest **AUC of 0.9960**, and the highest **F1 Score of 0.9512**. Its recall of **92.86%** also shows that it correctly identified most of the malignant cases in the test set.

### Decision Tree

The Decision Tree showed the lowest overall performance among the five models. It achieved **92.11% accuracy** and an **F1 Score of 0.8861**. Its recall of **83.33%** was lower than that of the better-performing models.

### KNN

KNN performed well on the test data, achieving **95.61% accuracy** and an **F1 Score of 0.9383**. Its performance was relatively close to Logistic Regression, although its AUC, recall and F1 Score were slightly lower.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved **100% precision**, meaning that all cases predicted as malignant were actually malignant in the test set. However, its recall was **83.33%**, indicating that some malignant cases were not identified by the model.

### Random Forest

Random Forest achieved **96.49% accuracy** and **100% precision**. It also obtained the highest **MCC score of 0.9258**. Its recall and F1 Score were slightly lower than those of Logistic Regression, although its overall performance was very close.

---

## G. Overall Winner

### Logistic Regression

Based on the comparison of all six evaluation metrics, **Logistic Regression** was selected as the overall best-performing model for this experiment.

Its performance on the test set was:

| Metric | Score |
|---|---:|
| Accuracy | 96.49% |
| AUC | 0.9960 |
| Precision | 97.50% |
| Recall | 92.86% |
| F1 Score | 95.12% |
| MCC | 0.9245 |

Logistic Regression achieved the highest **AUC, Recall and F1 Score**, while also matching Random Forest for the highest Accuracy.

Random Forest was a very close second and achieved the highest MCC score.

---

## H. Streamlit Application

A Streamlit-based web application was developed to demonstrate the trained classification models.

The application provides the following features:

1. Upload the test dataset in CSV format.
2. Select one of the five trained classification models.
3. View the performance of the selected model.
4. Display Accuracy, AUC, Precision, Recall, F1 Score and MCC.
5. Display the confusion matrix.
6. Display the classification report.
7. Display a summary of the predicted classes.

The application loads the saved trained models and preprocessing objects instead of retraining the models every time the application is opened.

---

## I. Project Structure

```text
ml-assignment2-cancer-classification/
│
├── Assignment2.ipynb
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
├── model_results.csv
├── wdbc.data
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    └── feature_names.pkl
