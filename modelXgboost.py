import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.map({'Yes': 1, 'No': 0})

    # Convert categorical features using one-hot encoding
    X = pd.get_dummies(X)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns  # Return feature names as well

# Train XGBoost model
def train_xgb_model(X, y, feature_names):
    # Train-validation-test split (70-20-10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Handle class imbalance using SMOTE
    # smote = SMOTE(random_state=42)
    # X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Define XGBoost model
    model = XGBClassifier(
        n_estimators=100,  
        max_depth=6,  
        learning_rate=0.1,  
        subsample=0.8,  
        colsample_bytree=0.8,  
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Performance metrics
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Print results
    print(f"Test Accuracy: {test_accuracy:.4f}") 
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC Score: {test_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # Feature Importance
    feature_importance = model.feature_importances_

    # Filter out features with 0 importance
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df[importance_df["Importance"] > 0]  # Keep only non-zero importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))  # Increased height for better label fitting
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance from XGBoost")
    
    # Rotate feature names to fit them
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability

    # Adjust layout to fit labels
    plt.tight_layout()

    plt.show()

    return model

# Run script
def main():
    filepath = "Thyroid_Diff.csv"  # Change to your actual dataset
    X, y, feature_names = load_data(filepath)
    model = train_xgb_model(X, y, feature_names)

if __name__ == "__main__":
    main()
