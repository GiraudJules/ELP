# Standard library imports
from abc import abstractmethod
from typing import Union, Dict, List

# Third party imports
import numpy as np
import pandas as pd

# Local applications imports
from src.classes.node import Node


class BaseTree:
    """
    Base class to build a Classification or Regression Tree.
    To build a child of this class and inherit the methods, need to implement the methods.
    """

    def __init__(self, min_sample_leaf, max_depth, min_sample_split):
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.classes = None

    def fit(self, dataframe: pd.DataFrame) -> Node:
        """
        - Retrieves the different classes from X_features and stores it into self.classes
        - Assign new value to self.node with self.create_node
        - Build the tree from new root and current_depth

        Args:
            X_features (np.array): X features from dataset
            y_features (np.array): y features from dataset

        Returns:
            self.root: new Node of the tree
        """
        # Retrieve the different classes from X_features and store it into self.classes
        self.classes = list(set(int(row[-1]) for row in dataframe.values))

        # Assign new value to self.node with self.create_node
        self.root = self.create_node(dataframe)

        # Build the tree from new root and current_depth
        self.build_tree(self.root, current_depth=0)

        # Return the root of the tree
        return self.root

    def predict(
        self, X_test: Union[Union[int, str], np.array]
    ) -> Union[Union[int, str], np.array]:
        """
        Predict class for whether:
        - Regression: a single int OR multiple int
        - Classification: a single str OR multiple str

        Args:
            X_test (Union[Union[int,str],np.array]): test features to predict on

        Returns:
            Union[Union[int, str], np.array]: whether a single int or str; or np.array

        """
        predictions = []

        for row in X_test.values:

            node = self.root

            while node.is_leaf is not True:

                if row[node.col_index] <= node.splitting_point:
                    node = node.left_child
                    continue

                if row[node.col_index] > node.splitting_point:
                    node = node.right_child

            predictions.append(node.value)

        return predictions

    def build_tree(self, node: Node, current_depth: int) -> None:
        """
        Build the tree recursively.

        Args:
            node (Node()): current node
            current_depth (int): current depth of the tree

        Returns:
            None
        """
        # Assert if current depth is less than max depth
        if current_depth < self.max_depth:
            # If left child is a list
            if node.left_child is not None and isinstance(node.left_child, list):
                # Instanciate a new node from left child
                left_node = self.create_node(node.left_child)
                node.left_child = left_node
                # While left node is not a leaf, build the tree
                if node.left_child.is_leaf is not True:
                    self.build_tree(node.left_child, current_depth + 1)

            # If right child is a list
            if node.right_child is not None and isinstance(node.right_child, list):
                # Instanciate a new node from right child
                right_node = self.create_node(node.right_child)
                node.right_child = right_node
                # While left node is not a leaf, build the tree
                if node.right_child.is_leaf is not True:
                    self.build_tree(node.right_child, current_depth + 1)

    def split_dataset(
        self,
        dataframe: pd.DataFrame,
        column_index: int,
        splitting_point: float,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into left and right datasets.

        Args:
            dataframe (pd.DataFrame): dataset to split
            column_index (int): index of the column to split
            splitting_point (float): value of the splitting point

        Returns:
            Dict[str:pd.DataFrame, str: pd.DataFrame]: dictionary with left and right datasets
        """
        # Instanciante empty dataframes
        left_child = pd.DataFrame()
        right_child = pd.DataFrame()

        # Iterate over dataset to split
        for i, val in enumerate(dataframe.values):

            # If value is less than or equal to splitting point, append to left child
            if dataframe.iloc[i, column_index] <= splitting_point:
                # Concatenate new value to dataframe
                left_child = pd.concat(
                    [left_child, dataframe.iloc[i, :]], axis=1, ignore_index=True
                )

            # If value is greater than splitting point, append to right child
            if dataframe.iloc[i, column_index] > splitting_point:
                # Concatenate new value to dataframe
                right_child = pd.concat(
                    [right_child, dataframe.iloc[i, :]], axis=1, ignore_index=True
                )

        # Return dictionary with left and right child
        return {"left": left_child.T, "right": right_child.T}

    @abstractmethod
    def create_node(self, data: list) -> Node:
        """
        Create a new node

        Args:
            data (np.array): X and y features for the left or right child of the node

        Returns:
            Node(): new node
        """
        raise NotImplementedError
