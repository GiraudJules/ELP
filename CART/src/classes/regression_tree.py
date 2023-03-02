import numpy as np
import pandas as pd

from typing import Union, List
from src.classes.base_tree import BaseTree
from src.classes.node import Node

class RegressionTree(BaseTree):
    def __init__(self, min_sample_leaf, min_sample_split) -> None:
        super().__init__(self, min_sample_leaf, min_sample_split)
        self.min_sample_leaf = min_sample_leaf
        self.min_sample_split = min_sample_split

    def create_node(self, data: list) -> Node:
        """
        Create a new node

        Args:
            data (list): List of X and y features for the left or right child of the node

        Returns:
            Node(): new node
        """
        node = Node(None)
        X, y = np.transpose(data[0]), data[1]

        ### Check if node has enough samples to be split again
        if len(data[1]) <= self.min_sample_split:
            node.is_leaf = True
            node.value = np.mean(data[1])
            return node
        
        risk_value = np.inf
        for col_index in X:
            X_sorted, y_sorted = self.sort_data(col_index, y)
            for row_index in range(len(X_sorted) - 1):
                value = ((X_sorted[row_index] + X_sorted[row_index + 1]) / 2)
                child = self.split_dataset(X_sorted, y_sorted, value)
                # Check the minimum number of samples required to be at a leaf node
                if ((len(child['left'][0]) >= self.min_sample_leaf) and (len(child['right'][0]) >= self.min_sample_leaf)):
                    candidate_risk_value = self.risk_regression(child)
                    print(candidate_risk_value)
                    if candidate_risk_value < risk_value:
                        print('True')
                        risk_value = candidate_risk_value
                        node.value = value
                        node.risk_value = candidate_risk_value
                        node.left_child = child['left']
                        node.right_child = child['right']

    def sort_data(self, X: np.array, y: np.array) -> tuple:
        """Sort data in order to try every split candidates

        Args:
            X (np.array): Xth feature
            y (np.array)

        Returns:
            tuple: Outputs sorted data
        """
        X_sorted, y_sorted = (list(t) for t in zip(*sorted(zip(X, y))))
        return X_sorted, y_sorted

    def mse(self, y: Union[np.array, List]) -> np.float64:
        """Compute Mean Square Error from the average of a given region R

        Args:
            y (Union[np.array, List]): Given region R (left or right)

        Returns:
            float: MSE
        """
        y_mean = np.mean(y)
        mse = np.square(y - y_mean).mean()
        return mse
    
    def risk_regression(self, child: dict) -> np.float64:
        """Compute the risk function for the regression from the two regions 

        Args:
            child (dict): dict representing left and right regions

        Returns:
            float: Risk value of two regions
        """
        print(child['left'])
        left_risk = self.mse(child['left'])
        right_risk =  self.mse(child['right'])

        risk_value = left_risk + right_risk
        return risk_value
    
