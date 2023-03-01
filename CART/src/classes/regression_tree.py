import numpy as np
import pandas as pd

from typing import Union, List
from base_tree import BaseTree
from node import Node

class RegressionTree(BaseTree):
    def __init__(self) -> None:
        super().__init__(self)

    def create_node(self, data: list) -> Node():
        """
        Create a new node

        Args:
            data (list): List of X and y features for the left or right child of the node

        Returns:
            Node(): new node
        """
        node = Node(None)
        X, y = data[0], data[1]

        ### Check if node has enough samples to be split again
        if len(data[1]) <= self.min_sample_split:
            node.is_leaf = True
            node.value = np.mean(data[1])
            return node
        
        risk_value = np.inf
        for col_index in X:
            sorted_data = self.sort_data(col_index, y) 
            for row_index in range(len(sorted_data[0]) - 1):
                value = (sorted_data[0][row_index] + sorted_data[0][row_index + 1] / 2)
                child = self.split_dataset(col_index, y, value)
                # Check the minimum number of samples required to be at a leaf node
                if (len(child['left']) >= self.min_sample_leaf & len(child['right']) >= self.min_sample_leaf):
                    candidate_risk_value = self.risk_regression(child)
                    if candidate_risk_value < risk_value:
                        risk_value = candidate_risk_value
                        node.value = value
                        node.risk_value = candidate_risk_value
                        node.left_child = child['left']
                        node.right_child = child['right']

    def sort_data(X: np.array, y: np.array) -> tuple:
        """Sort data in order to try every split candidates

        Args:
            X (np.array): Xth feature
            y (np.array)

        Returns:
            tuple: Outputs sorted data
        """
        data = np.vstack((X, y))
        data = np.sort(data, axis=-1)
        return data

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
        left_risk = self.mse(child['left'])
        right_risk =  self.mse(child['right'])

        risk_value = left_risk + right_risk
        return risk_value
    
