# wine_quality_predictor_colab.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from google.colab import files
import io

# Step 1: Upload and clean dataset
def load_and_clean_data():
    uploaded = files.upload()
    for file_name in uploaded:
        df = pd.read_csv(io.BytesIO(uploaded[file_name]))

    # Impute missing values with mean
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
def train_and_save_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'wine_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved.")

# Step 4: Predict on user input
def predict_sample(model, scaler):
    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol']

    print("\nEnter the following chemical attributes:")
    input_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        input_data.append(value)

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    confidence = np.max(model.predict_proba(input_scaled))

    result = "Good" if prediction else "Not Good"
    print(f"\nPrediction: {result} (Confidence: {confidence:.2f})")

# Step 5: Run in Colab
def main():
    df = load_and_clean_data()
    (X_train, X_test, y_train, y_test), scaler = prepare_data(df)
    train_and_save_model(X_train, X_test, y_train, y_test)

    model = joblib.load('wine_model.pkl')
    scaler = joblib.load('scaler.pkl')
    predict_sample(model, scaler)

if __name__ == "__main__":
    main()
