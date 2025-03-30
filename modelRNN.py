import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn import metrics
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def thyroid_cancer_recurrence_model(filepath):
  # Load data
  df = pd.read_csv(filepath)
  
  # Encode categorical features using LabelEncoder for target and OneHotEncoder for input features
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

  # Split into train and test sets (80-20 split)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # RNN model
  model = keras.Sequential()
  model.add(keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(64, activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(32, activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(1, activation="sigmoid"))

  # Compile model
  model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

  # Train model with validation set
  model.fit(X_train, y_train, validation_split=0.2, epochs=25, shuffle=True, batch_size=16)

  # Evaluate model on test set
  loss, accuracy = model.evaluate(X_test, y_test)
  
  # Generate predictions and classification report
  y_pred = (model.predict(X_test) > 0.5).astype(int)
  report = metrics.classification_report(y_test, y_pred, output_dict=True)

  # Display results
  print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
  print(metrics.classification_report(y_test, y_pred))

  #return model, report

def main():
  filepath = "Thyroid_Diff.csv"
  print(thyroid_cancer_recurrence_model(filepath))

if __name__ == "__main__":
  main()
