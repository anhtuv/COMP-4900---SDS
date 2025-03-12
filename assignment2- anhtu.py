from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# Name this file assignment2.py when you submit


# Question 1
# A function that implements a keras model with the sequential API following the provided description
def sequential_model():

  # Creating a sequential model
  model = keras.Sequential(name="sequential_model")
  
  # Adding the layers as specified:
  # Hidden layer with 16 neurons with an input layer of 8 neurons and ReLU activation function
  model.add(keras.layers.Dense(16, input_dim = 8, activation = "relu"))

  # Hidden layer with 12 neurons and ReLU activation function
  model.add(keras.layers.Dense(12, activation = "relu"))

  # Hidden layer with 8 neurons and ReLU activation function
  model.add(keras.layers.Dense(8, activation = "relu"))

  # Output layer with 4 neurons with softmax activation
  model.add(keras.layers.Dense(4, activation = "softmax"))
  
  # Compiling model with stochastic gradient descent optimizer, categorical cross entropy loss, and accuracy as the metric
  model.compile(optimizer = keras.optimizers.SGD(),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
  )
  
  # Uncomment to view model architecture
  # print(model.summary())
  # A keras model
  return model


# Question 2
# A function that implements a keras model with the functional API following the provided description
def functional_model():

  # Input layer with 4 neurons
  input = keras.layers.Input(shape=(4,))

  # Input layer is passed into the first hidden layer with 8 neurons and ReLU activation function
  hidden_layer_1 = keras.layers.Dense(8, activation = "relu")(input)
  
  # First hidden layer is passed into the second hidden layer of 2 neurons and ReLU activation function
  hidden_layer_2 = keras.layers.Dense(2, activation = "relu")(hidden_layer_1)

  # Passing the second hidden layer into both output layers with 2 neurons and softmax activation function  
  output_layer_1 = keras.layers.Dense(2, activation = "softmax")(hidden_layer_2)
  output_layer_2 = keras.layers.Dense(2, activation = "softmax")(hidden_layer_2)
  
  # Creating the model and specifying input and outputs
  model = keras.models.Model(inputs = input, outputs = [output_layer_1, output_layer_2])
  
  # Compiling model with stochastic gradient descent optimizer, sum of BCE loss function, and accuracy as the metric
  model.compile(optimizer = keras.optimizers.SGD(),
    loss = sum_bce,
    metrics = ["accuracy"]
  )
  
  # Uncomment to view model architecture
  # print(model.summary())
  
  # A keras model
  return model

# Calculate the sum of binary cross entropy loss
def sum_bce(y, y_hat):
  loss = keras.losses.binary_crossentropy(y, y_hat)
  return tf.reduce_sum(loss)


# Question 3
# A function that creates a keras model to predict whether a patient has recurrence of thryroid cancer
# model is a trained keras model for predicting whether a a patient has recurrence of thryroid cancer during a follow-up period
def thyroid_cancer_recurrence_model(filepath):
  # Load data
  data = pd.read_csv(filepath)
  x = data.iloc[:, :-1]
  y = data.iloc[:, -1]

  # Preprocess data
  quantitative_data  = x.columns[x.dtypes == "int64"]
  #print(numerical_cols)
  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(x[quantitative_data])

  qualitative_data = x.columns[x.dtypes == "object"]
  #print(categorical_cols)
  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(x[qualitative_data])

  y = y.map({"No": 0, "Yes": 1, })

  # Split into train and test sets
  x_train, x_test, y_train, y_test = train_test_split(np.hstack((quantitative_preprocessed, qualitative_preprocessed)), y, test_size=0.2, shuffle=True)

  # Build model
  model = keras.Sequential()
  model.add(keras.layers.Dense(64, input_shape=(x_train.shape[1],), activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(64, activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(32, activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(1, activation="sigmoid"))

  # Train model
  model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy, metrics=['accuracy'])
  model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=25, shuffle=True, batch_size=16)

  # validation_performance is the performance of the model on a validation set
  validation_performance = model.evaluate(x_test, y_test)
  print(validation_performance)

  return model, validation_performance


# Question 4a
def experiment_neurons(filepath):
  # Load data
  df = pd.read_csv(filepath)
  x = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  # Preprocess data
  quantitative_data  = x.columns[x.dtypes == "int64"]
  #print(numerical_cols)
  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(x[quantitative_data])

  qualitative_data = x.columns[x.dtypes == "object"]
  #print(categorical_cols)
  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(x[qualitative_data])

  y = y.map({"No": 0, "Yes": 1, })

  x_train, x_test, y_train, y_test = train_test_split(np.hstack((quantitative_preprocessed, qualitative_preprocessed)), y, test_size=0.2, shuffle=True)

  neurons = [16, 32, 64, 128]
  val_accuracies = []

  for n in neurons:
    model = keras.Sequential()
    model.add(keras.layers.Dense(n, input_shape=(x_train.shape[1],), activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(n, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(n, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy, metrics=['accuracy'])
    # history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, shuffle=True, batch_size=16, verbose=0)

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=16, verbose=0)
    val_accuracies.append(history.history['val_accuracy'][-1])

  plt.plot(neurons, val_accuracies, marker='.')
  plt.xlabel('Number of Neurons')
  plt.ylabel('Validation Accuracy')
  plt.title('Performance on the validation set as a function of the number of neurons in each hidden layer')
  plt.show()


# Question 4b
def experiment_layers(filepath):
  # Load data
  df = pd.read_csv(filepath)
  x = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  # Preprocess data
  quantitative_data  = x.columns[x.dtypes == "int64"]
  #print(numerical_cols)
  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(x[quantitative_data])

  qualitative_data = x.columns[x.dtypes == "object"]
  #print(categorical_cols)
  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(x[qualitative_data])

  y = y.map({"No": 0, "Yes": 1, })

  x_train, x_test, y_train, y_test = train_test_split(np.hstack((quantitative_preprocessed, qualitative_preprocessed)), y, test_size=0.2, shuffle=True)

  layers = [1, 2, 3, 4]
  val_accuracies = []

  for l in layers:
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(x_train.shape[1],), activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    for _ in range(l - 1):
      model.add(keras.layers.Dense(64, activation="relu"))
      model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy, metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=16, verbose=0)
    val_accuracies.append(history.history['val_accuracy'][-1])

  plt.plot(layers, val_accuracies, marker='.')
  plt.xlabel('Number of Hidden Layers')
  plt.ylabel('Validation Accuracy')
  plt.title('Performance on the validation set as a function of the number of hidden layers')
  plt.show()


# Question 4c
def experiment_epochs(filepath):
  # Load data
  df = pd.read_csv(filepath)
  x = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  # Preprocess data
  quantitative_data  = x.columns[x.dtypes == "int64"]
  #print(numerical_cols)
  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(x[quantitative_data])

  qualitative_data = x.columns[x.dtypes == "object"]
  #print(categorical_cols)
  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(x[qualitative_data])

  y = y.map({"No": 0, "Yes": 1, })

  x_train, x_test, y_train, y_test = train_test_split(np.hstack((quantitative_preprocessed, qualitative_preprocessed)), y, test_size=0.2, shuffle=True)

  epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  val_accuracies = []

  for e in epochs:
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(x_train.shape[1],), activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy, metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=e, batch_size=16, verbose=0)
    val_accuracies.append(history.history['val_accuracy'][-1])

  plt.plot(epochs, val_accuracies, marker='.')
  plt.xlabel('Number of Epochs')
  plt.ylabel('Validation Accuracy')
  plt.title('Performance on the validation set as a function of the number of epochs')
  plt.show()

# Question 4d
# number of neurons in each hidden layer: combination of 64 and 32
# number of hidden layers: 2
# number of epochs: 25
def optimal_model(filepath):
  # Load data
  df = pd.read_csv(filepath)
  x = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  # Preprocess data
  quantitative_data  = x.columns[x.dtypes == "int64"]
  #print(numerical_cols)
  scaler = StandardScaler()
  quantitative_preprocessed = scaler.fit_transform(x[quantitative_data])

  qualitative_data = x.columns[x.dtypes == "object"]
  #print(categorical_cols)
  encoder = OneHotEncoder(sparse_output=False)
  qualitative_preprocessed = encoder.fit_transform(x[qualitative_data])

  y = y.map({"No": 0, "Yes": 1, })

  x_train, x_test, y_train, y_test = train_test_split(np.hstack((quantitative_preprocessed, qualitative_preprocessed)), y, test_size=0.2, shuffle=True)
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

  model = keras.Sequential()
  model.add(keras.layers.Dense(64, input_shape=(x_train.shape[1],), activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(64, activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(32, activation="relu"))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Dense(1, activation="sigmoid"))

  model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy, metrics=['accuracy'])

  model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=16, verbose=0)
  train_performance = model.evaluate(x_train, y_train, verbose=0)
  val_performance = model.evaluate(x_val, y_val, verbose=0)
  test_performance = model.evaluate(x_test, y_test, verbose=0)

  print(f"Training Performance: {train_performance}")
  print(f"Validation Performance: {val_performance}")
  print(f"Test Performance: {test_performance}")

def main():
  filepath = "Thyroid_Diff.csv"
  #sequential_model()
  #functional_model()
  #thyroid_cancer_recurrence_model(filepath)
  #experiment_neurons(filepath)
  #experiment_layers(filepath)
  #experiment_epochs(filepath)
  optimal_model(filepath)

  
if __name__ == "__main__":
  main()