import numpy as np
import pandas as pd 
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE, mutual_info_classif, chi2

def mutual_information(x_train, y_train):
    mutual_info = mutual_info_classif(x_train, y_train)
    mutual_info = pd.Series(mutual_info, index=x_train.columns).to_dict()
    return mutual_info
# higher scores == most important

def lasso_reg(x_train, y_train):
    pipeline = Pipeline([('scaler',StandardScaler()), ('model',Lasso())])
    search = GridSearchCV(pipeline, {'model__alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error")
    search.fit(x_train,y_train)
    search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    return {feature: float(coef) for feature, coef in zip(x_train.columns, importance)}
# higher scores == most important

def chi_square(x_train, y_train, x_test, qualitative_preprocessed):
    x_train_cat = x_train.loc[:, qualitative_preprocessed.columns]
    x_test_cat = x_test.loc[:, qualitative_preprocessed.columns]
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(x_train_cat, y_train)
    chi2_selector.transform(x_test_cat)
    scores = chi2_selector.scores_
    return {feature: float(score) for feature, score in zip(x_train_cat.columns, scores)}
# higher scores == most important

def rec_feat_elim(x_train, y_train):
    model = LogisticRegression()
    rfe = RFE(estimator=model, n_features_to_select=1)
    x_train_rfe = rfe.fit_transform(x_train, y_train)
    model.fit(x_train_rfe, y_train)
    return {feature: int(rank) for feature, rank in zip(x_train.columns, rfe.ranking_)}
# 1 == most important