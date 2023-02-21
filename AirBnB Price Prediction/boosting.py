#####################
#This Python file contains helper functions to be used to create Boosting models.
######################

import numpy as np
import pandas as pd
from utils import eval 
from xgboost import XGBRegressor
from utils import search_parameters, splits, eval


#----------------------------------------------------------------------------------------#
#                                    XGBoost Regressor                                   #
#----------------------------------------------------------------------------------------#

def create_xgboost(X, y, grid_search=False, params=None, metrics=None):
    '''
    This function allows to create, evaluate and make predictions an XGBoost model for a regression task.
    Input:
        - Input data
        - Output labels
        - Boolean to indicate whether or not to perform a GridSearchCV
        - Dictionary of parameters to be used to initialized a model
        - List of metrics to be passed as input in the evaluation helper function
    Output:
        - Fitted model
        - Test output labels
        - Prediction labels made by fitted model 
    '''

    #creating training and test sets using split helper function
    X_train, y_train, X_test, y_test = splits(X, y, test_size=0.2)

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
    

    #PREDICTIONS:
    y_pred = xgb_reg.predict(X_test)


    #MODEL EVALUATION: using eval() helper function
    if metrics == None:
        xgb_scores = eval(xgb_reg, X, y)
    else:
        xgb_scores = eval(xgb_reg, X, y, scores=metrics)
    
    return xgb_reg, y_test, y_pred
