import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt

def preprocess_data(data):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Preprocess qualitative data (one-hot encoding)
    qualitative_data = x.columns[x.dtypes == "object"].values
    encoder = OneHotEncoder(sparse_output=False)
    qualitative_preprocessed = pd.DataFrame(
        encoder.fit_transform(x[qualitative_data]), 
        columns=encoder.get_feature_names_out(qualitative_data)
    )

    x_preprocessed = pd.concat([qualitative_preprocessed], axis=1)

    # Map target variable ("Yes" and "No" to 1 and 0)
    y = y.map({"No": 0, "Yes": 1})
    
    return x_preprocessed, y

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)

    # Probabilities for log loss & AUC
    y_train_prob = rf_model.predict_proba(X_train)[:, 1]
    y_val_prob = rf_model.predict_proba(X_val)[:, 1]
    y_test_prob = rf_model.predict_proba(X_test)[:, 1]

    # Accuracy scores
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # F1 Scores
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Log Loss
    train_loss = log_loss(y_train, y_train_prob)
    val_loss = log_loss(y_val, y_val_prob)
    test_loss = log_loss(y_test, y_test_prob)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # AUC Score
    auc = roc_auc_score(y_test, y_test_prob)

    # Print results
    print("random forest model")
    print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")
    print("Confusion Matrix:\n", cm)
    print("AUC Score:", auc)

    # # Feature Importance
    # feature_names = pd.get_dummies(pd.read_csv("Thyroid_Diff.csv").iloc[:, :-1]).columns
    # feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
    # feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # # Plot top features
    # plt.figure(figsize=(12, 8))
    # plt.barh(feature_importance["Feature"][:10], feature_importance["Importance"][:10])  # Top 10 features
    # plt.xlabel("Importance Score")
    # plt.ylabel("Feature")
    # plt.title("Feature Importance (Random Forest)")
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    #plt.show()
    return test_acc

def main():
    filepath = "Thyroid_Diff.csv"
    X, y = preprocess_data(filepath)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    train_model(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()
