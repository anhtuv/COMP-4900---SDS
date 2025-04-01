import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, confusion_matrix
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def thyroid_cancer_recurrence_model(filepath):
  # Load data
  df = pd.read_csv(filepath)
  
  # Encode categorical features
  labelencoder = LabelEncoder()
  df['Recurred'] = labelencoder.fit_transform(df['Recurred'])

  quantitative_data = df.select_dtypes(include=['int64', 'float64']).drop('Recurred', axis=1)
  qualitative_data = df.select_dtypes(include=['object'])

  # Scale numerical features
  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(quantitative_data)

  # One-hot encode categorical features
  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(qualitative_data)

  # Combine processed data
  X = np.hstack((quantitative_preprocessed, qualitative_preprocessed))
  y = df['Recurred']

  # Train-validation-test split (70-20-10)
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

  # RNN model
  model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation="sigmoid")
  ])

  # Compile model
  model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

  # Train model
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, shuffle=True, batch_size=16, verbose=0)

  # Evaluate model
  train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
  val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
  test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

  # Predictions
  y_train_pred = (model.predict(X_train) > 0.5).astype(int)
  y_val_pred = (model.predict(X_val) > 0.5).astype(int)
  y_test_pred = (model.predict(X_test) > 0.5).astype(int)

  # Probabilities for AUC calculation
  y_test_proba = model.predict(X_test)

  # Metrics
  train_f1 = f1_score(y_train, y_train_pred)
  val_f1 = f1_score(y_val, y_val_pred)
  test_f1 = f1_score(y_test, y_test_pred)
  
  train_loss = log_loss(y_train, y_train_pred)
  val_loss = log_loss(y_val, y_val_pred)
  test_loss = log_loss(y_test, y_test_pred)

  # Print results
  print(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Loss: {train_loss:.4f}")
  print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Validation Loss: {val_loss:.4f}")
  print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Loss: {test_loss:.4f}")
  
  # Confusion Matrix
  cm = confusion_matrix(y_test, y_test_pred)
  print("Confusion Matrix:\n", cm)
  
  # AUC Score
  auc = roc_auc_score(y_test, y_test_proba)
  print("AUC Score:", auc)

  # # Generate predictions and classification report
  # y_pred = (model.predict(X_test) > 0.5).astype(int)
  # report = metrics.classification_report(y_test, y_pred, output_dict=True)

  # # Display results
  # print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
  # print(metrics.classification_report(y_test, y_pred))

def main():
  filepath = "Thyroid_Diff.csv"
  thyroid_cancer_recurrence_model(filepath)

if __name__ == "__main__":
  main()
