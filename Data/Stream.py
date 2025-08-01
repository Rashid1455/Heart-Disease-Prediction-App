import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# Streamlit page settings
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("💓 Heart Disease Prediction App")

# Load and preprocess data
def load_and_preprocess(df):
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric)
    X = df.drop('condition', axis=1)
    y = (df['condition'] > 0).astype(int)
    return X, y

# Train Random Forest model
def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Evaluate the model and show ROC & classification report
def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"], output_dict=True)

    st.subheader("🔍 ROC AUC Score")
    st.write(f"AUC: **{auc:.3f}**")

    st.subheader("📋 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# Plot permutation feature importance
def plot_feature_importance(model, X_test, y_test):
    st.subheader("📊 Feature Importance (Permutation)")
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = perm.importances_mean
    indices = np.argsort(importances)[::-1]
    features = X_test.columns

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(features)), importances[indices])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features[indices], rotation=90)
    ax.set_ylabel("Mean decrease in score")
    ax.set_title("Permutation Feature Importances")
    st.pyplot(fig)

# Streamlit UI
uploaded_file = st.file_uploader("📂 Upload the Heart Disease Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File successfully loaded. Here's a preview:")
        st.dataframe(df.head())

        X, y = load_and_preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = train_rf(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        plot_feature_importance(model, X_test, y_test)

    except Exception as e:
        st.error(f"❌ Error loading or processing the file: {e}")
else:
    st.info("👆 Please upload a CSV file to get started.")
