# wine_quality_streamlit_app.py

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and clean dataset
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("winequality-red.csv")
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

# Step 2: Prepare dataset for modeling
def prepare_data(df):
    df['is_good'] = df['quality'] >= 7
    X = df.drop(['quality', 'is_good'], axis=1)
    y = df['is_good']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

# Step 3: Train and save model
def train_and_save_model(X_train, X_test, y_train, y_test, scaler):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.subheader("Model Training Results")
    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 2))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))
    joblib.dump(model, 'wine_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    st.success("Model and scaler saved successfully.")

# Step 4: Predict from input
def predict_sample(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled))
    return prediction, confidence

# Step 5: Streamlit UI
def main():
    st.set_page_config(page_title="Wine Quality Classifier", layout="centered")
    st.title("🍷 Red Wine Quality Classifier")
    st.write("This app predicts whether a red wine is of good quality based on its chemical attributes.")

    with st.expander("🔧 Train Model (Uses Internal Dataset)", expanded=False):
        df = load_and_clean_data()
        (X_train, X_test, y_train, y_test), scaler = prepare_data(df)
        train_and_save_model(X_train, X_test, y_train, y_test, scaler)

    st.markdown("---")
    st.subheader("🧪 Enter Wine Sample Details")

    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol']

    input_data = []
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        col = col1 if i % 2 == 0 else col2
        value = col.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.2f")
        input_data.append(value)

    if st.button("🔍 Predict Quality"):
        try:
            model = joblib.load('wine_model.pkl')
            scaler = joblib.load('scaler.pkl')
            prediction, confidence = predict_sample(model, scaler, input_data)
            result = "🍷 Good Quality Wine" if prediction else "⚠️ Not Good Quality Wine"
            st.success(f"**Prediction:** {result}\n\n**Confidence:** {confidence:.2f}")
        except FileNotFoundError:
            st.error("Model not found. Please run training first.")

if __name__ == "__main__":
    main()
