import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, confusion_matrix
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, confusion_matrix, log_loss, roc_auc_score
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

def load_data(filepath):
  df = pd.read_csv(filepath)
  
  labelencoder = LabelEncoder()
  df['Recurred'] = labelencoder.fit_transform(df['Recurred'])

  quantitative_data = df.select_dtypes(include=['int64', 'float64']).drop('Recurred', axis=1)
  qualitative_data = df.select_dtypes(include=['object'])

  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(quantitative_data)

  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(qualitative_data)

  X = np.hstack((quantitative_preprocessed, qualitative_preprocessed))
  y = df['Recurred']
  
  return X, y

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
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

  # Plot feature importance and metrics
  # lgb.plot_importance(model)
  # plt.savefig('importance.png')
  # lgb.plot_metric(model)
  # plt.savefig('metric.png')

def main():
  filepath = "Thyroid_Diff.csv"
  X, y = load_data(filepath)
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
  train_model(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
  main()
