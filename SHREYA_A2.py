# Name this file assignment2.py when you submit

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, metrics
import matplotlib.pyplot as plt

# A function that implements a keras model with the sequential API following the provided description
def sequential_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(8,)))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# A function that implements a keras model with the functional API following the provided description
def functional_model():
    my_input_layer = keras.Input(shape=(4,))
    
    my_hidden_layer_1 = keras.layers.Dense(8, activation="relu")
    hidden1 = my_hidden_layer_1(my_input_layer)
    
    my_hidden_layer_2 = keras.layers.Dense(2, activation="relu")
    hidden2 = my_hidden_layer_2(hidden1)
    
    my_output_layer = keras.layers.Dense(2, activation="softmax")
    my_output = my_output_layer(hidden2)

    my_functional_model = keras.Model(
        inputs=my_input_layer, 
        outputs=my_output, 
        name="q2"
    )

    my_functional_model.compile(
        optimizer='sgd', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    return my_functional_model

# A function that creates a keras model to predict whether a patient has recurrence of thryroid cancer
def thyroid_cancer_recurrence_model(filepath="Thyroid_Diff.csv"):
  df = pd.read_csv(filepath)
  
  # no null data (fire dataset), no filtering needed
  
  # encoding string data:
  labelencoder = preprocessing.LabelEncoder()
  for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = labelencoder.fit_transform(df[col])
  
  # splitting data
  X = df.drop('Recurred', axis=1)
  y = df['Recurred']

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
  
  def perform_classification(model): # results
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    # print(metrics.classification_report(y_test, y_pred)) UNPRINT FOR NEAT RESULTS
    return report

  model = keras.Sequential()
  model.add(keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],))) # chose neurons = features
  model.add(keras.layers.Dense(16, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid')) # sigmoid for bin class

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  validation_performance = perform_classification(model)
  return model, validation_performance