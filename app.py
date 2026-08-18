import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer ML Classifier",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------
# Load saved models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),
        "KNN": joblib.load(
            "model/knn.pkl"
        ),
        "Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),
        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")

    return models, scaler, feature_names


models, scaler, feature_names = load_models()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🧬 Breast Cancer Classification")
st.subheader("Machine Learning Model Evaluation Dashboard")

st.write(
    "This application compares five classification models "
    "using the Breast Cancer Wisconsin Diagnostic dataset."
)

st.divider()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

st.sidebar.info(
    "Upload the test CSV generated during the ML experiment."
)

# ---------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully.")

    st.write("### Uploaded Data")
    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Check required columns
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in test_data.columns
    ]

    if missing_features:

        st.error(
            "The uploaded file is missing required feature columns: "
            + ", ".join(missing_features)
        )

    elif "diagnosis" not in test_data.columns:

        st.error(
            "The uploaded test file must contain the diagnosis column."
        )

    else:

        # -------------------------------------------------
        # Prepare test data
        # -------------------------------------------------

        X_uploaded = test_data[feature_names]
        y_actual = test_data["diagnosis"]

        model = models[selected_model]

        # Scale only models trained using scaled data
        if selected_model in [
            "Logistic Regression",
            "KNN"
        ]:

            X_model_input = scaler.transform(X_uploaded)

        else:

            X_model_input = X_uploaded

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        predictions = model.predict(X_model_input)

        probabilities = model.predict_proba(
            X_model_input
        )[:, 1]

        # -------------------------------------------------
        # Evaluation metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_actual,
            predictions
        )

        auc = roc_auc_score(
            y_actual,
            probabilities
        )

        precision = precision_score(
            y_actual,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_actual,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_actual,
            predictions,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_actual,
            predictions
        )

        # -------------------------------------------------
        # Metrics display
        # -------------------------------------------------

        st.subheader(
            f"Performance: {selected_model}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

        st.subheader("Confusion Matrix")

        matrix = confusion_matrix(
            y_actual,
            predictions
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Actual Benign",
                "Actual Malignant"
            ],
            columns=[
                "Predicted Benign",
                "Predicted Malignant"
            ]
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

        # -------------------------------------------------
        # Classification report
        # -------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_actual,
            predictions,
            target_names=[
                "Benign",
                "Malignant"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # -------------------------------------------------
        # Prediction summary
        # -------------------------------------------------

        st.subheader("Prediction Summary")

        predicted_counts = pd.Series(
            predictions
        ).value_counts()

        summary_df = pd.DataFrame({
            "Class": [
                "Benign",
                "Malignant"
            ],
            "Predicted Count": [
                predicted_counts.get(0, 0),
                predicted_counts.get(1, 0)
            ]
        })

        st.dataframe(
            summary_df,
            use_container_width=True
        )

else:

    st.info(
        "Upload test_data.csv to evaluate the selected model."
    )
