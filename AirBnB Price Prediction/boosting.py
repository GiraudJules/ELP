#####################
#This Python file contains helper functions to be used to create Boosting models.
#Author: Amandine Allmang
######################

import numpy as np
import pandas as pd
from utils import eval 
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from utils import search_parameters, splits, eval


#----------------------------------------------------------------------------------------#
#                                    XGBoost Regressor                                   #
#----------------------------------------------------------------------------------------#

def create_xgboost(X, y, grid_search=False, params=None, metrics=None, y_preproc=False):
    '''
    This function allows to create, evaluate and make predictions an XGBoost model for a regression task.
    Input:
        - X: Input data
        - y: Output labels
        - grid_search: Boolean to indicate whether or not to perform a GridSearchCV
        - params: Dictionary of parameters to be used to initialized a model
        - metrics: List of metrics to be passed as input in the evaluation helper function
        - y_preproc: a boolean to select if normalization of y is required
    Output:
        - xgb_reg: Fitted model
        - y_test: Test output labels
        - y_pred: Prediction labels made by fitted model 
    '''

    #creating training and test sets using split helper function
    X_train, X_test, y_train, y_test = splits(X, y, test_size=0.2, y_preproc=y_preproc)

    #CREATING MODEL:
    #Option 1: if grid_search = True is passed: launch a grid search
    if grid_search == True:
        #initializing a XGBRegressor model
        xgb_init = XGBRegressor()
        #dictionary of parameters to test
        xgb_params = { 'n_estimators': [50,100, 150, 200],
                    'gamma': [0, 0.5, 1, 2],
                    'max_depth': [3, 5, 6, 7],
                    'learning_rate': [1.0, 0.1, 0.01]
                    }
        #using helper function to launch grid search
        xgb_best_params = search_parameters(xgb_init,xgb_params,X_train,y_train)
        #initializing an XXGBRegressor with the best parameters found
        xgb_reg = XGBRegressor(**xgb_best_params)
    
    #Option 2: if no parameters are given, initialize a model with default parameters
    elif params == None:
        xgb_reg = XGBRegressor()
    
    #Option 3: if parameters are given and no GridSearch is required, initialize a model with the given parameters
    else:
        #initialise model with given parameters
        xgb_reg = XGBRegressor(**params)


    #MODEL EVALUATION: using eval() helper function
    if metrics == None:
        xgb_scores = eval(xgb_reg, X, y)
    else:
        xgb_scores = eval(xgb_reg, X, y, scores=metrics)
    
    #FITTING MODEL:
    xgb_reg.fit(X_train, y_train)

    #PREDICTIONS:
    y_pred = xgb_reg.predict(X_test)

    return xgb_reg, y_test, y_pred


#----------------------------------------------------------------------------------------#
#                                   Adaboost Regressor                                   #
#----------------------------------------------------------------------------------------#

def create_adaboost(X, y, grid_search=False, params=None, metrics=None, y_preproc=False):
    '''
    This function allows to create, evaluate and make predictions an Adaboost model for a regression task.
    Input:
        - X: Input data
        - y: Output labels
        - grid_search: Boolean to indicate whether or not to perform a GridSearchCV
        - params: Dictionary of parameters to be used to initialized a model
        - metrics:List of metrics to be passed as input in the evaluation helper function
        - y_preproc: a boolean to select if normalization of y is required
    Output:
        - ada_reg: Fitted model
        - y_test: Test output labels
        - y_pred: Prediction labels made by fitted model 
    '''

    #creating training and test sets using split helper function
    X_train, X_test, y_train, y_test = splits(X, y, test_size=0.2, y_preproc=y_preproc)

    #CREATING MODEL:
    #Option 1: if grid_search = True is passed: launch a grid search
    if grid_search == True:
        #initializing a XGBRegressor model
        ada_init = AdaBoostRegressor()
        #dictionary of parameters to test
        adaboost_params = {'n_estimators': [10, 50, 100, 150, 200],
                            'learning_rate': [0.1, 0.01, 1.0]
                            }
        #using helper function to launch grid search
        ada_best_params = search_parameters(ada_init, adaboost_params, X_train, y_train)
        #initializing an AdaBoostRegressor with the best parameters found
        ada_reg = AdaBoostRegressor(**ada_best_params)
    
    #Option 2: if no parameters are given, initialize a model with default parameters
    elif params == None:
        ada_reg = AdaBoostRegressor()
    
    #Option 3: if parameters are given and no GridSearch is required, initialize a model with the given parameters
    else:
        #initialise model with given parameters
        ada_reg = AdaBoostRegressor(**params)


    #MODEL EVALUATION: using eval() helper function
    if metrics == None:
        ada_scores = eval(ada_reg, X, y)
    else:
        ada_scores = eval(ada_reg, X, y, scores=metrics)
    
    #FITTING MODEL:
    ada_reg.fit(X_train, y_train)

    #PREDICTIONS:
    y_pred = ada_reg.predict(X_test)

    return ada_reg, y_test, y_pred


#----------------------------------------------------------------------------------------#
#                              Gradient Boosting Regressor                               #
#----------------------------------------------------------------------------------------#

def create_gradboost(X, y, grid_search=False, params=None, metrics=None, y_preproc=False):
    '''
    This function allows to create, evaluate and make predictions a Gradient Boosting model for a regression task.
    Input:
        - X: Input data
        - y: Output labels
        - grid_search: Boolean to indicate whether or not to perform a GridSearchCV
        - params: Dictionary of parameters to be used to initialized a model
        - metrics: List of metrics to be passed as input in the evaluation helper function
        - y_preproc: a boolean to select if normalization of y is required
    Output:
        - grad_reg: Fitted model
        - y_test: Test output labels
        - y_pred: Prediction labels made by fitted model 
    '''

    #creating training and test sets using split helper function
    X_train, X_test, y_train, y_test = splits(X, y, test_size=0.2, y_preproc=y_preproc)

    #CREATING MODEL:
    #Option 1: if grid_search = True is passed: launch a grid search
    if grid_search == True:
        #initializing a XGBRegressor model
        grad_init = GradientBoostingRegressor()
        #dictionary of parameters to test
        gradboost_params = {'learning_rate': [0.1, 0.01],
                            'n_estimators': [50, 100, 150],
                            'min_samples_leaf': [1,2,3,5],
                            'min_samples_split': [2,3,5],
                            'max_depth': [3,5,7]
                            }
        #using helper function to launch grid search
        grad_best_params = search_parameters(grad_init, gradboost_params, X_train, y_train)
        #initializing an GradientBoostingRegressor with the best parameters found
        grad_reg = GradientBoostingRegressor(**grad_best_params)
    
    #Option 2: if no parameters are given, initialize a model with default parameters
    elif params == None:
        grad_reg = GradientBoostingRegressor()
    
    #Option 3: if parameters are given and no GridSearch is required, initialize a model with the given parameters
    else:
        #initialise model with given parameters
        grad_reg = GradientBoostingRegressor(**params)


    #MODEL EVALUATION: using eval() helper function
    if metrics == None:
        grad_scores = eval(grad_reg, X, y)
    else:
        grad_scores = eval(grad_reg, X, y, scores=metrics)
    
    #FITTING MDOEL
    grad_reg.fit(X_train, y_train)
    
    #PREDICTIONS:
    y_pred = grad_reg.predict(X_test)

    return grad_reg, y_test, y_pred
