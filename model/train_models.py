import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data.prep_data import prepare_training_data

def train_models(data):
    """
    Train multiple machine learning models on stock data and compare accuracy.
    """

    # Prepare training data
    X, y = prepare_training_data(data)

    if len(X) < 50:
        print("⚠️ Not enough data to train models.")
        return None

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42) # change depth from 7 --> 10, estimators from 150 --> 300
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    # ✅ Logistic Regression Model
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)

    # ✅ Gradient Boosting Model (XGBoost)
    gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=5, random_state=42) # estimators from 100 --> 300, learning rate from 0.05 --> 0.01
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)

    # ✅ Support Vector Machine (SVM)
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)

    print("\n📊 Model Performance:")
    print(f"✅ Random Forest Accuracy: {rf_acc:.4f}")
    print(f"✅ Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"✅ Gradient Boosting Accuracy: {gb_acc:.4f}")
    print(f"✅ SVM Accuracy: {svm_acc:.4f}")

    return {"RF": rf_model, "LR": log_model, "GB": gb_model, "SVM": svm_model}
