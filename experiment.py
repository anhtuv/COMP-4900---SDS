from feature_selection import mutual_information, lasso_reg, rec_feat_elim, chi_square
from sklearn.model_selection import train_test_split
import modelLogisticReg as logreg
import modelSVM as svm
import modelKNN as knn
import modelRNN as rnn
import modelRandomForest as random_forest
import modelLightGradientBoost as lg_boost
import modelXgboost as xg_boost
# import matplotlib as plt
import pandas as pd
# import numpy as np

feature_selection_methods = {
        "Mutual Information": mutual_information,
        "Lasso": lasso_reg,
        "Recursive Feature Elimination": rec_feat_elim,
        "Chi-Square": chi_square
    }
    
classification_models = {
        "Light Gradient Boosting": lg_boost,
        "Logistic Regression": logreg,
        "K-Nearest Neighbour": knn,
        "Random Forest": random_forest,
        "RNN": rnn,
        "SVM": svm,
        "XGBoost": xg_boost
    }


def load_data(filepath):
    data = pd.read_csv(filepath)
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]
    labels = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s']
    data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels).astype(str)
    return data

def run_experiments(model, data):
    results = {}
    
    x, y = model.preprocess_data(data)
    
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33, random_state=42)

    for feat_sel in feature_selection_methods:
        results[feat_sel] = []
        features = feature_selection_methods[feat_sel](x_train, y_train)
        for x in range(0, len(features)):
            selected_features = features[:x]
            x_train_selected = x_train[selected_features]
            x_val_selected = x_val[selected_features]
            x_test_selected = x_test[selected_features]
            results[feat_sel].append(model.train_model(x_train_selected, y_train, x_val_selected, y_val, x_test_selected, y_test))
    return results
    
#graph for a specific model
    #x == x features
    #y == performance measure
    #colours == feature selection model
    
    
data = load_data("Thyroid_Diff.csv")

for model in classification_models:
    run_experiments(classification_models[model], data)