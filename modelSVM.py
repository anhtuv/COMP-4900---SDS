import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, hinge_loss
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.inspection import permutation_importance

def preprocess_data(data):
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]

  y = y.map({'Yes': 1, 'No': 0})

  X = pd.get_dummies(X)

  # Scale data using MinMaxScaler (scales between 0 and 1)
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)

  return X_scaled, y

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
  model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
  model.fit(X_train, y_train)

  # Predictions and probabilities
  y_train_pred = model.predict(X_train)
  y_val_pred = model.predict(X_val)
  y_test_pred = model.predict(X_test)

  y_test_proba = model.predict_proba(X_test)[:, 1]

  # Accuracy scores
  train_acc = accuracy_score(y_train, y_train_pred)
  val_acc = accuracy_score(y_val, y_val_pred)
  test_acc = accuracy_score(y_test, y_test_pred)

  # F1 Scores
  train_f1 = f1_score(y_train, y_train_pred)
  val_f1 = f1_score(y_val, y_val_pred)
  test_f1 = f1_score(y_test, y_test_pred)

  # Hinge loss
  y_train_hinge = np.where(y_train == 1, 1, -1)
  y_val_hinge = np.where(y_val == 1, 1, -1)
  y_test_hinge = np.where(y_test == 1, 1, -1)

  # Predict decision function for hinge loss calculation
  train_decision = model.decision_function(X_train)
  val_decision = model.decision_function(X_val)
  test_decision = model.decision_function(X_test)

  train_loss = hinge_loss(y_train_hinge, train_decision)
  val_loss = hinge_loss(y_val_hinge, val_decision)
  test_loss = hinge_loss(y_test_hinge, test_decision)

  # Train an SVM with a linear kernel for feature selection
  svm_linear = SVC(kernel="linear", C=1, class_weight="balanced")

  # Use RFE to select top features
  selector = RFE(svm_linear, n_features_to_select=5)
  selector.fit(X_train, y_train)
  feature_names = pd.get_dummies(pd.read_csv("Thyroid_Diff.csv").iloc[:, :-1]).columns
  selected_features = feature_names[selector.support_]

  #print("Selected Features:", selected_features)

  # Feature importance using permutation importance
  result = permutation_importance(model, X_train, y_train, scoring="accuracy", n_repeats=10, random_state=42)
  importance_scores = result.importances_mean

  # Convert to DataFrame for visualization
  importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance_scores})
  importance_df = importance_df.sort_values(by="Importance", ascending=False)

  # Plot the top features
  plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
  plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])  # Top 10 features
  plt.xlabel("Importance Score")
  plt.ylabel("Feature")
  plt.title("Feature Importance (Permutation Method)")

  # Rotate feature names for better readability
  plt.xticks(rotation=45, ha="right")

  # Invert y-axis to show the most important feature at the top
  plt.gca().invert_yaxis()

  # Adjust layout to prevent label overlap
  plt.tight_layout()

  #plt.show()

  # Print results
  print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")
  print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
  print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")

  cm = confusion_matrix(y_test, y_test_pred)
  print("Confusion Matrix:\n", cm)

  auc = roc_auc_score(y_test, y_test_proba)
  print("AUC Score:", auc)

def main():
  filepath = "Thyroid_Diff.csv"
  X, y = preprocess_data(filepath)
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
  train_model(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
  main()
