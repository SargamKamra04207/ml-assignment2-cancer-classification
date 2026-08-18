# Breast Cancer Classification using Machine Learning

## A. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for identifying whether a breast tumor is **benign or malignant**.

Five classification algorithms were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

The models were evaluated using the following performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

A Streamlit application was also developed to allow users to upload test data, select a machine learning model, and view its evaluation results.

## B. Dataset Description

The **Breast Cancer Wisconsin Diagnostic dataset** from the UCI Machine Learning Repository was used for this experiment.

The dataset contains 569 observations, 30 numerical predictive features, and one binary target variable (diagnosis).

The target variable represents whether the tumor is:

| Value | Meaning   |
| 0     | Benign    |
| 1     | Malignant |

The dataset was divided into **80% training data and 20% testing data** using stratified sampling.

| Dataset      | Number of Samples |
| Training Set |         455       |
| Testing Set  |         114       |
| Total        |         569       |

The original `id` column was removed because it is an identifier and does not provide useful predictive information.
The dataset contained **no missing values**.
Feature scaling using `StandardScaler` was applied to **Logistic Regression and KNN**, while the **Decision Tree, Gaussian Naive Bayes and Random Forest** models were trained using the original feature values.

### Preprocessing and Modelling Approach

The dataset was first inspected for missing values and duplicate or non-predictive identifier information. The `id` column was removed before modelling.

The `diagnosis` values were converted into binary numerical labels, where 0 represents a benign case and 1 represents a malignant case.

The data was divided into training and testing sets using an 80:20 stratified split. StandardScaler was used for Logistic Regression and KNN because these models are sensitive to feature scale. Decision Tree, Gaussian Naive Bayes and Random Forest were trained using the unscaled feature values.

All five models were evaluated on the same held-out test set.

## C. GitHub Repository

https://github.com/SargamKamra04207/ml-assignment2-cancer-classification

## D. Models Used

The following five classification models were implemented:

|         ML Model      |               Type            |
| Logistic Regression   | Linear Classification         |
| Decision Tree         | Tree-Based Classification     |
| KNN                   | Distance-Based Classification |
| Gaussian Naive Bayes  | Probabilistic Classification  |
| Random Forest         | Ensemble Classification       |

### Model Comparison

The models were evaluated on the same test dataset using six performance metrics.

|       ML Model Name     | Accuracy |   AUC  | Precision | Recall | F1 Score |   MCC  |
| Logistic Regression     | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512   | 0.9245 |
| Decision Tree           | 0.9211   | 0.9448 | 0.9459    | 0.8333 | 0.8861   | 0.8299 |
| KNN                     | 0.9561   | 0.9825 | 0.9744    | 0.9048 | 0.9383   | 0.9058 |
| Gaussian Naive Bayes    | 0.9386   | 0.9934 | 1.0000    | 0.8333 | 0.9091   | 0.8715 |
| Random Forest (Ensemble)| 0.9649   | 0.9940 | 1.0000    | 0.9048 | 0.9500   | 0.9258 |

## E. Model-wise Observations

|             ML Model         |                                                                                                 Observation                                                                                                                                                                               |
| **Logistic Regression**      | Logistic Regression gave the best overall performance in this experiment. It achieved **96.49% accuracy**, the highest **AUC of 0.9960**, and the highest **F1 Score of 0.9512**. Its recall of **92.86%** also indicates that it identified most of the malignant cases in the test set. |
| **Decision Tree**            | The Decision Tree produced the lowest overall performance among the five models, with **92.11% accuracy** and an **F1 Score of 0.8861**. Its recall was **83.33%**, which was lower than the stronger-performing models.                                                                  |
| **KNN**                      | KNN performed well with **95.61% accuracy** and an **F1 Score of 0.9383**. Its performance was close to Logistic Regression, although its AUC, recall and F1 Score were slightly lower.                                                                                                   |
| **Gaussian Naive Bayes**     | Gaussian Naive Bayes achieved **100% precision**, meaning that all cases predicted as malignant were actually malignant in the test set. However, its recall was **83.33%**, indicating that it missed some malignant cases.                                                              |
| **Random Forest (Ensemble)** | Random Forest achieved **96.49% accuracy** and **100% precision**. It also obtained the highest **MCC score of 0.9258**. Its recall and F1 Score were marginally lower than Logistic Regression.                                                                                          |


## F. Overall Winner

### 🏆 Logistic Regression
**Logistic Regression** is considered the overall winner for this dataset.

It achieved:
- **Accuracy:** 96.49%
- **AUC:** 0.9960
- **Precision:** 97.50%
- **Recall:** 92.86%
- **F1 Score:** 95.12%
- **MCC:** 0.9245

Logistic Regression achieved the **highest AUC, Recall and F1 Score**, while also matching Random Forest for the highest Accuracy.

Random Forest was a very close second and achieved the highest MCC score.

## G. Streamlit Application

A Streamlit-based web application was developed to demonstrate the trained machine learning models.

The application provides the following functionality:

1. Upload the test dataset as a CSV file.
2. Select one of the five trained classification models.
3. View model performance metrics.
4. View Accuracy, AUC, Precision, Recall, F1 Score and MCC.
5. View the confusion matrix.
6. View the classification report.
7. View the prediction summary.

The application uses the saved trained models and preprocessing objects rather than retraining the models each time the application is opened.

## H. Project Structure

ml-assignment2-cancer-classification/
│
├── Assignment2.ipynb
├── app.py
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
