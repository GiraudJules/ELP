import numpy as np
import pandas as pd

from typing import Union, List
from src.classes.base_tree import BaseTree
from src.classes.node import Node


class RegressionTree(BaseTree):
    def __init__(self, min_sample_leaf, min_sample_split, max_depth) -> None:
        super().__init__(self, min_sample_leaf, min_sample_split)
        self.min_sample_leaf = min_sample_leaf
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def create_node(self, data) -> Node:
        """
        Create a new node

        Args:
            data (list): List of X and y features for the left or right child of the node

        Returns:
            Node(): new node
        """
        node = Node(None)

        ### Check if node has enough samples to be split again
        if len(data) <= self.min_sample_split:
            node.is_leaf = True
            node.predicted_value = data[data.columns[-1]].mean()
            return node

        risk_value = np.inf

        for col_index in range(len(data.columns) - 2):

            data_sorted = data.sort_values(by=data.columns[col_index])
            for row_index in range(len(data_sorted) - 1):

                splitting_value = (
                    data_sorted.iloc[row_index][col_index]
                    + data_sorted.iloc[row_index + 1][col_index]
                ) / 2

                child = self.split_dataset(data_sorted, col_index, splitting_value)

                if (len(child["left"]) >= self.min_sample_leaf) and (
                    len(child["right"]) >= self.min_sample_leaf
                ):

                    candidate_risk_value = self.risk_regression(child)

                    # If it is current lowest MSE, we update the node
                    if candidate_risk_value < risk_value:

                        risk_value = candidate_risk_value

                        ## Update the node
                        # Which value to separate data
                        node.splitting_point = splitting_value
                        # Index of the feature X
                        node.column_index = col_index

                        # Set of predicted value for this node
                        node.predicted_value = data[data.columns[-1]].mean()

                        # Set of X/y which go to left and right
                        node.left_region = child["left"]
                        node.right_region = child["right"]
                        # We set that the node is not a leaf
                        node.is_leaf = False

                # else:
                #
                #     # Set of X/y which go to left and right
                #     node.left_region = child["left"]
                #     node.right_region = child["right"]
                #     node.is_leaf = True
                #

        return node

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

    def mse(self, y: pd.DataFrame) -> np.float64:
        """Compute Mean Square Error from the average of a given region R

        Args:
            y (Union[np.array, List]): Given region R (left or right)

        Returns:
            float: MSE
        """
        y_mean = y.mean()
        mse = np.square(y - y_mean).mean()
        return mse

    def risk_regression(self, child: dict) -> np.float64:
        """Compute the risk function for the regression from the two regions

        Args:
            child (dict): dict representing left and right regions

        Returns:
            float: Risk value of two regions
        """

        df_left = child["left"]
        df_right = child["right"]

        if len(df_left) > 0:
            y_left = df_left.iloc[:, -1]
            left_risk = self.mse(y_left)
        else:
            left_risk = 0

        if len(df_right) > 0:
            y_right = df_right.iloc[:, -1]
            right_risk = self.mse(y_right)
        else:
            right_risk = 0

        total_length = len(df_left) + len(df_right)
        risk_value = (
            len(df_left) / total_length * left_risk
            + len(df_right) / total_length * right_risk
        )
        return risk_value
