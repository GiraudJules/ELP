# Standard library imports
from abc import abstractmethod
from typing import Union, Dict

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
        """ Build the tree from new root and current_depth

        Args:
            dataframe (pd.DataFrame): Train dataset to fit the DecisionTree on

        Returns:
            self.root: New root Node of the tree
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
        self, test_data: pd.DataFrame
    ) -> Union[Union[int, str], np.array]:
        """
        Predict class for whether:
        - Regression: a single int OR multiple int
        - Classification: a single str OR multiple str

        Args:
            X_test (pd.DataFrame): Test dataset to predict on

        Returns:
            Union[Union[int, str], np.array]: whether a single int or str; or np.array

        """
        predictions = []

        for row in test_data.values:

            node = self.root

            while node.is_leaf is not True:

                if row[node.column_index] <= node.splitting_point:
                    node = node.left_child_node
                    continue

                if row[node.column_index] > node.splitting_point:
                    node = node.right_child_node

            predictions.append(node.predicted_value)

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
        if (current_depth < self.max_depth) and (node.is_leaf is False):
            # If left child is a list
            if node.left_region is not None:
                assert isinstance(
                    node.left_region, pd.DataFrame
                ), "Node child must be of type <pd.DataFrame>"

                # Instanciate a new node from left child
                left_node = self.create_node(node.left_region)
                node.left_child_node = left_node

                # While left node is not a leaf, build the tree
                if left_node.is_leaf is not True:
                    self.build_tree(left_node, current_depth + 1)

            # If right child is a list
            if node.right_region is not None:
                assert isinstance(
                    node.right_region, pd.DataFrame
                ), "Node child must be of type <pd.DataFrame>"

                # Instanciate a new node from right child
                right_node = self.create_node(node.right_region)
                node.right_child_node = right_node

                # While left node is not a leaf, build the tree
                if right_node.is_leaf is not True:
                    self.build_tree(right_node, current_depth + 1)
        else:
            node.is_leaf = True

    @staticmethod
    def split_dataset(
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
        left_region = pd.DataFrame()
        right_region = pd.DataFrame()

        # Iterate over dataset to split
        for i, _ in enumerate(dataframe.values):

            # If value is less than or equal to splitting point, append to left child
            if dataframe.iloc[i, column_index] <= splitting_point:
                # Concatenate new value to dataframe
                left_region = pd.concat(
                    [left_region, dataframe.iloc[i, :]], axis=1, ignore_index=True
                )

            # If value is greater than splitting point, append to right child
            elif dataframe.iloc[i, column_index] > splitting_point:
                # Concatenate new value to dataframe
                right_region = pd.concat(
                    [right_region, dataframe.iloc[i, :]], axis=1, ignore_index=True
                )

        # Return dictionary with left and right child
        return {"left": left_region.T, "right": right_region.T}

    @abstractmethod
    def create_node(self, data: pd.DataFrame) -> Node:
        """
        Abstract method to be implemented in specific child classes.
        Creates a new node from left or right region (dataframe).

        Args:
            data (pd.DataFrame): X and y features for the left or right child of the node

        Raises:
            NotImplementedError: abstract method
        """
        raise NotImplementedError
