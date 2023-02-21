import numpy as np
import pandas as pd
from utils import eval 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#----------------------------------------------------------------------------------------#
#                                    Decision Tree                                       #
#----------------------------------------------------------------------------------------#

def create_a_decision_tree(X, y, params=None):
    if params == None:
        decision_tree = DecisionTreeRegressor()
    else:
        decision_tree = DecisionTreeRegressor(**params)

    # Evaluating performance 
    decision_tree_scores = eval(decision_tree, X, y)
    return decision_tree_scores


#----------------------------------------------------------------------------------------#
#                                      Random Forest                                     #
#----------------------------------------------------------------------------------------#

def create_a_random_forest(X, y, params=None):
    if params == None:
        random_forest = RandomForestRegressor()
    else:
        random_forest = RandomForestRegressor(**params)

    # Evaluating performance 
    random_forest_scores = eval(random_forest, X, y)
    return random_forest_scores