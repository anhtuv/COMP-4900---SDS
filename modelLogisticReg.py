import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.map({'Yes': 1, 'No': 0})  # Assuming binary classification
    X = pd.get_dummies(X)  # One-hot encoding for categorical features
    
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns

def train_logistic_regression_model(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    log_reg = LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced', random_state=42)
    log_reg.fit(X_train, y_train)

    # Predictions
    y_train_pred = log_reg.predict(X_train)
    y_val_pred = log_reg.predict(X_val)
    y_test_pred = log_reg.predict(X_test)

    # Accuracy scores
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # F1 Scores
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Log Loss (Loss function for training, validation, and test)
    y_train_prob = log_reg.predict_proba(X_train)[:, 1]  
    y_val_prob = log_reg.predict_proba(X_val)[:, 1]  
    y_test_prob = log_reg.predict_proba(X_test)[:, 1]  

    train_loss = log_loss(y_train, y_train_prob)
    val_loss = log_loss(y_val, y_val_prob)
    test_loss = log_loss(y_test, y_test_prob)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # AUC Score
    auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

    # Print results
    print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")
    print("Confusion Matrix:\n", cm)
    print("AUC Score:", auc)

    return log_reg

def select_important_features(X, y, feature_names, k=5):
    # Use mutual_info_classif which works better for Logistic Regression
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    
    return X_new, selected_features

def main():
    filepath = "Thyroid_Diff.csv"
    X, y, feature_names = load_data(filepath)
    X_selected, selected_features = select_important_features(X, y, feature_names, k=5)
    LogisticReg = train_logistic_regression_model(X_selected, y)

if __name__ == "__main__":
    main()
