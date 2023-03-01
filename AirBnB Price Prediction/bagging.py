'''
A python file to create different types of bagging methods.
Author: Marie Bouvard
'''
import numpy as np
import pandas as pd
from utils import splits, search_parameters, eval 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor



#----------------------------------------------------------------------------------------#
#                                    Decision Tree                                       #
#----------------------------------------------------------------------------------------#

def create_a_decision_tree(X, y, grid_search=False, params=None, metrics=None, verbose=1):
    '''
    This function creates and evaluate a decision tree. Optionally, we can perform a GridSearch to 
    obtain the best parameters or given specific parameters for our tree.

    Input: 
        - X: our data
        - y: the corresponding labels, here the price
        - params: a dictionnary containing paramters to be passed to a decision tree.
        - metrics: a list of metrics to be used for the evaluation of our model.

    Output:
        - decision_tree: the created decision tree, 
        - y_test: the labels associated to the test set
        - y_pred: the predicted values on a test set
    '''

    X_train, X_test, y_train, y_test = splits(X, y)

    if grid_search == True:
        # Find best parameters with GridSearch Function, and evaluate
        dt_params = {'criterion': ['poisson', 'absolute_error', 'squared_error', 'friedman_mse'], 
                    'min_samples_split':[2,5,10,25],
                    'min_samples_leaf':[2,5,10,25],
                    'max_depth':[2,5,10,25,50,75,100,150]
                    }

        # Obtain best parameters from grid search function
        dt_best_params = search_parameters(DecisionTreeRegressor(), dt_params, X_train, y_train, verbose=verbose)
        decision_tree = DecisionTreeRegressor(**dt_best_params)
        
    elif params == None:
        # If no parameters are given create a default decision tree
        decision_tree = DecisionTreeRegressor()

    else:
        # Creating a decision tree with a given parameter dictionnary
        decision_tree = DecisionTreeRegressor(**params)

    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)

    # Evaluating performance 
    if metrics == None:
        decision_tree_scores = eval(decision_tree, X, y)
    else: 
        decision_tree_scores = eval(decision_tree, X, y, scores=metrics)
        
    return decision_tree, y_test, y_pred

#----------------------------------------------------------------------------------------#
#                                      Random Forest                                     #
#----------------------------------------------------------------------------------------#

def create_a_random_forest(X, y, grid_search=False, params=None, metrics=None, verbose=1):
    '''
    This function creates and evaluate a Random Forest. Optionally, we can perform a GridSearch to 
    obtain the best parameters or given specific parameters for our forest.

    Input: 
        - X: our data
        - y: the corresponding labels, here the price
        - params: a dictionnary containing paramters to be passed to a random forest.
        - metrics: a list of metrics to be used for the evaluation of our model.

    Output:
        - random_forest: the created random forest, 
        - y_test: the labels associated to the test set
        - y_pred: the predicted values on a test set
    '''

    X_train, X_test, y_train, y_test = splits(X, y)

    if grid_search == True:
        # Create a Random Forest, find best parameters with GridSearch Function, and evaluate
        rf_params = {'criterion': ['poisson', 'absolute_error', 'squared_error', 'friedman_mse'],
            'n_estimators': [5,25,50,75,100,500], 
            'min_samples_split':[2,5,10,25],
            'min_samples_leaf':[2,5,10,25],
            'max_depth':[2,5,10,25,50,75,100,150],
            'oob_score': [True, False]
            }

        # Obtain best parameters from grid search function
        rf_best_params = search_parameters(RandomForestRegressor(), rf_params, X_train, y_train, verbose=verbose)
        random_forest_best = RandomForestRegressor(**rf_best_params)
        
    elif params == None:
        # If no parameters are given create a default Random Forest
        random_forest = RandomForestRegressor()

    else:
        # Creating a Random Forest with a given parameter dictionnary
        random_forest = RandomForestRegressor(**params)

    # Evaluating performance 
    if metrics==None:
        random_forest_scores = eval(random_forest, X, y)
    else:
        random_forest_scores = eval(random_forest, X, y, scores=metrics)

    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)

    return random_forest, y_test, y_pred


#----------------------------------------------------------------------------------------#
#                           Extremely Randomized Random Forest                           #
#----------------------------------------------------------------------------------------#

def create_an_extremely_randomized_forest(X, y, grid_search=False, params=None, metrics=None, verbose=1):
    '''
    This function creates and evaluate an extremely randomized Random Forest. Optionally, we can 
    perform a GridSearch to obtain the best parameters or given specific parameters for our forest.

    Input: 
        - X: our data
        - y: the corresponding labels, here the price
        - params: a dictionnary containing paramters to be passed to a random forest.
        - metrics: a list of metrics to be used for the evaluation of our model.

    Output:
        - extra_random_forest: the created extremely random forest, 
        - y_test: the labels associated to the test set
        - y_pred: the predicted values on a test set
    '''

    X_train, X_test, y_train, y_test = splits(X, y)

    if grid_search == True:
        # Create a Random Forest, find best parameters with GridSearch Function, and evaluate
        extra_rf_params = {'criterion': ['poisson', 'absolute_error', 'squared_error', 'friedman_mse'],
            'n_estimators': [5,25,50,75,100,500], 
            'min_samples_split':[2,5,10,25],
            'min_samples_leaf':[2,5,10,25],
            'max_depth':[2,5,10,25,50,75,100,150],
            'oob_score': [True, False]
            }

        # Obtain best parameters from grid search function
        extra_rf_best_params = search_parameters(ExtraTreesRegressor(), extra_rf_params, X_train, y_train, verbose=1)
        random_forest_best = ExtraTreesRegressor(**extra_rf_best_params)
        
    elif params == None:
        # If no parameters are given create a default Random Forest
        extra_random_forest = ExtraTreesRegressor()

    else:
        # Creating a Random Forest with a given parameter dictionnary
        extra_random_forest = ExtraTreesRegressor(**params)

    # Evaluating performance 
    if metrics==None:
        extra_random_forest_scores = eval(extra_random_forest, X, y)
    else:
        extra_random_forest_scores = eval(extra_random_forest, X, y, scores=metrics)

    y_pred = extra_random_forest.predict(X_test)

    return extra_random_forest, y_test, y_pred


#----------------------------------------------------------------------------------------#
#                           Extremely Randomized Random Forest                           #
#----------------------------------------------------------------------------------------#

def create_an_extremely_randomized_forest(X, y, grid_search=False, params=None, metrics=None, verbose=1):
    '''
    This function creates and evaluate an extremely randomized Random Forest. Optionally, we can 
    perform a GridSearch to obtain the best parameters or given specific parameters for our forest. 

    Input: 
        - X: our data
        - y: the corresponding labels, here the price
        - params: a dictionnary containing paramters to be passed to a random forest.
        - metrics: a list of metrics to be used for the evaluation of our model.

    Output:
        - extra_random_forest: the created extremely random forest, 
        - y_test: the labels associated to the test set
        - y_pred: the predicted values on a test set 
    '''

    X_train, y_train, X_test, y_test = splits(X, y)

    if grid_search == True:
        # Create a Random Forest, find best parameters with GridSearch Function, and evaluate
        extra_rf_params = {'criterion': ['poisson', 'absolute_error', 'squared_error', 'friedman_mse'],
            'n_estimators': [5,25,50,75,100,500], 
            'min_samples_split':[2,5,10,25],
            'min_samples_leaf':[2,5,10,25],
            'max_depth':[2,5,10,25,50,75,100,150],
            'oob_score': [True, False]
            }

        # Obtain best parameters from grid search function
        extra_rf_best_params = search_parameters(ExtraTreesRegressor(), extra_rf_params, X_train, y_train, verbose=1)
        random_forest_best = ExtraTreesRegressor(**extra_rf_best_params)
        
    elif params == None:
        # If no parameters are given create a default Random Forest
        extra_random_forest = ExtraTreesRegressor()

    else:
        # Creating a Random Forest with a given parameter dictionnary
        extra_random_forest = ExtraTreesRegressor(**params)
 
    # Evaluating performance 
    if metrics==None:
        extra_random_forest_scores = eval(extra_random_forest, X, y)
    else:
        extra_random_forest_scores = eval(extra_random_forest, X, y, scores=metrics)

    extra_random_forest.fit(X_train, y_train)
    y_pred = extra_random_forest.predict(X_test) 

    return extra_random_forest, y_test, y_pred
