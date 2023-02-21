import numpy as np
import pandas as pd
from utils import eval 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#----------------------------------------------------------------------------------------#
#                                    Decision Tree                                       #
#----------------------------------------------------------------------------------------#

def create_a_decision_tree(X, y, params=None, metrics = None):
    '''
    This function creates and evaluate a decision tree. Optionally, we can perform a GridSearch to 
    obtain the best parameters or given specific parameters for our tree.

    Input: 
        - X: our data
        - y: the corresponding labels, here the price
        - params: a dictionnary containing paramters to be passed to a decision tree.
        - metrics: a list of metrics to be used for the evaluation of our model.

    Output:
        - A dictonnary containing the performance of our decision tree over different metrics 
    '''

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

    return decision_tree_scores


#----------------------------------------------------------------------------------------#
#                                      Random Forest                                     #
#----------------------------------------------------------------------------------------#

def create_a_random_forest(X, y, params=None, metrics=None):
    '''
    This function creates and evaluate a Random Forest. Optionally, we can perform a GridSearch to 
    obtain the best parameters or given specific parameters for our forest.

    Input: 
        - X: our data
        - y: the corresponding labels, here the price
        - params: a dictionnary containing paramters to be passed to a random forest.
        - metrics: a list of metrics to be used for the evaluation of our model.

    Output:
        - A dictonnary containing the performance of our random forest over different metrics 
    '''

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

    return random_forest_scores