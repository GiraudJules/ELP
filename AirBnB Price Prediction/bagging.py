import numpy as np
import pandas as pd
from utils import preprocessing, eval #search_parameters_bagging,
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  GridSearchCV


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


