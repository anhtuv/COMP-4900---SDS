import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    y = y.map({'Yes': 1, 'No': 0})
    X = pd.get_dummies(X)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y#, X.columns

def select_important_features(X, y, feature_names, k=5):
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    
    print("Selected Features:", selected_features)
    return X_selected, selected_features

def train_knn_model(X_train, y_train, X_val, y_val, X_test, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = knn_model.predict(X_train)
    y_val_pred = knn_model.predict(X_val)
    y_test_pred = knn_model.predict(X_test)

    # Probabilities for log loss & AUC
    y_train_prob = knn_model.predict_proba(X_train)[:, 1]
    y_val_prob = knn_model.predict_proba(X_val)[:, 1]
    y_test_prob = knn_model.predict_proba(X_test)[:, 1]

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
    print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")
    print("Confusion Matrix:\n", cm)
    print("AUC Score:", auc)

    return knn_model

def main():
    filepath = "Thyroid_Diff.csv"
    # X, y, feature_names = load_data(filepath)
    X, y = load_data(filepath)

    # Select important features
   # X_selected, selected_features = select_important_features(X, y, feature_names, k=5)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    train_knn_model(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()
