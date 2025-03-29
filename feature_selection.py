import numpy as np
import pandas as pd 
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE, mutual_info_classif, chi2

def mutual_information(x_train, y_train):
    mutual_info = mutual_info_classif(x_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_train.columns
    sel_five_cols = SelectKBest(mutual_info_classif, k=5)
    sel_five_cols.fit(x_train, y_train)
    return np.array(x_train.columns[sel_five_cols.get_support()])

def lasso_reg(x_train, y_train):
    pipeline = Pipeline([('scaler',StandardScaler()), ('model',Lasso())])
    search = GridSearchCV(pipeline, {'model__alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error")
    search.fit(x_train,y_train)
    search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    return np.array(x_train.columns[importance > 0])

def chi_square(x_train, y_train, x_test, qualitative_preprocessed):
    x_train_cat = x_train.loc[:, qualitative_preprocessed.columns]
    x_test_cat = x_test.loc[:, qualitative_preprocessed.columns]
    chi2_selector = SelectKBest(chi2, k=5)
    chi2_selector.fit_transform(x_train_cat, y_train)
    chi2_selector.transform(x_test_cat)
    return np.array(x_train_cat.columns[chi2_selector.get_support()])

def rec_feat_elim(x_train, y_train):
    model = LogisticRegression()
    rfe = RFE(estimator=model, n_features_to_select=5)
    x_train_rfe = rfe.fit_transform(x_train, y_train)
    model.fit(x_train_rfe, y_train)
    return np.array(x_train.columns[rfe.support_])