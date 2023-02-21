import numpy as np
import pandas as pd
from utils import splits, eval 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#----------------------------------------------------------------------------------------#
#                                    Decision Tree                                       #
#----------------------------------------------------------------------------------------#

def create_a_decision_tree(X, y, params=None, metrics=None):
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
    _, _, X_test, y_test = splits(X, y)

    if params == None:
        # If no parameters are given create a default decision tree
        decision_tree = DecisionTreeRegressor()

    else:
        # Creating a decision tree with a given parameter dictionnary
        decision_tree = DecisionTreeRegressor(**params)

    # Evaluating performance 
    if metrics == None:
        decision_tree_scores = eval(decision_tree, X, y)
    else: 
        decision_tree_scores = eval(decision_tree, X, y, scores=metrics)

    y_pred = decision_tree.predict(X_test)
        
    return decision_tree, y_test, y_pred

#----------------------------------------------------------------------------------------#
#                                      Random Forest                                     #
#----------------------------------------------------------------------------------------#

def create_a_random_forest(X, y,  params=None, metrics=None):
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
    _, _, X_test, y_test = splits(X, y)

    if params == None:
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

    y_pred = random_forest.predict(X_test)

    return random_forest, y_test, y_pred