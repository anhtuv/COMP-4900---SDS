import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import hinge_loss
from imblearn.over_sampling import SMOTE

def load_data(filepath):
  data = pd.read_csv(filepath)
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]
  
  # Convert categorical features using one-hot encoding
  X = pd.get_dummies(X)
  
  # Scale data
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  
  return X_scaled, y

def train_svm_model(X, y):
  # Split into training and test sets (70-20-10 split)
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

  # Handle class imbalance using SMOTE
  # smote = SMOTE(random_state=42)
  # X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

  # Train SVM with class weight to handle imbalance
  model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True)
  model.fit(X_train, y_train) #bal?

  # Predictions and probabilities
  y_train_pred = model.predict(X_train) #bal
  y_val_pred = model.predict(X_val)
  y_test_pred = model.predict(X_test)
  
  y_test_proba = model.predict_proba(X_test)[:, 1]

  # Accuracy scores
  train_acc = accuracy_score(y_train, y_train_pred) #bal
  val_acc = accuracy_score(y_val, y_val_pred)
  test_acc = accuracy_score(y_test, y_test_pred)

  # Hinge loss (convert labels to {-1, 1} for hinge loss)
  y_train_hinge = np.where(y_train == 1, 1, -1) #bal
  y_val_hinge = np.where(y_val == 1, 1, -1)
  y_test_hinge = np.where(y_test == 1, 1, -1)

  # Predict decision function for hinge loss calculation
  train_decision = model.decision_function(X_train) #bal
  val_decision = model.decision_function(X_val)
  test_decision = model.decision_function(X_test)

  train_loss = hinge_loss(y_train_hinge, train_decision)
  val_loss = hinge_loss(y_val_hinge, val_decision)
  test_loss = hinge_loss(y_test_hinge, test_decision)

  # Print results
  print(f"Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
  print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
  print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

  cm = confusion_matrix(y_test, y_test_pred)
  print("Confusion Matrix:\n", cm)

  auc = roc_auc_score(y_test, y_test_proba)
  print("AUC Score:", auc)

def main():
  filepath = "Thyroid_Diff.csv"
  X, y = load_data(filepath)
  model = train_svm_model(X, y)
  print(model)


if __name__ == "__main__":
  main()
