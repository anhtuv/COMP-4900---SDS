import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):#, feature_names):
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict on training, validation, and test sets
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_proba)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_loss = log_loss(y_val, y_val_proba)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_proba)
    
    # Print results
    print("xgboost model")
    print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")

    print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")
    # print(f"Test Accuracy: {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:\n", cm)
    
    # AUC Score
    auc = roc_auc_score(y_test, y_test_proba)
    print("AUC Score:", auc)

    # # Feature Importance
    # feature_importance = model.feature_importances_
    
    # # Filter out features with 0 importance
    # importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    # importance_df = importance_df[importance_df["Importance"] > 0]  
    # importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # # Plot feature importance
    # plt.figure(figsize=(12, 8)) 
    # sns.barplot(x="Importance", y="Feature", data=importance_df)
    # plt.xlabel("Feature Importance Score")
    # plt.ylabel("Features")
    # plt.title("Feature Importance from XGBoost")
    
    # plt.xticks(rotation=45, ha="right") 
    # plt.tight_layout()

    # plt.show()

    return test_acc

def main():
    filepath = "Thyroid_Diff.csv"
    X, y = preprocess_data(filepath)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    train_model(X_train, y_train, X_val, y_val, X_test, y_test) #feature_names)

if __name__ == "__main__":
    main()
