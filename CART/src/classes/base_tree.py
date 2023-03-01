# Standard library imports
from abc import ABC, abstractmethod
from typing import Union, Dict, List

# Third party imports
import numpy as np

# Local applications imports
from node import Node


class BaseTree:
    """
    Base class to build a Classification or Regression Tree.
    To build a child of this class and inherit the methods, need to implement the methods.
    """

    def __init__(self, min_sample_leaf, max_depth, min_sample_split):
        super().__init__()
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.classes = None

    def fit(self, X_features: np.array, y_features: np.array) -> Node():
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
        self.classes = y_features
        data = [X_features, y_features]
        self.root = self.create_node(data)
        self.build_tree(self.root, current_depth=0)

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

        raise NotImplementedError

    def build_tree(self, node: Node(), current_depth: int) -> None:
        """
        Build the tree recursively.

        Args:
            node (Node()): current node
            current_depth (int): current depth of the tree

        Returns:
            None
        """
        raise NotImplementedError

    def split_dataset(
        self,
        X_features: np.array,
        y_features: np.array,
        splitting_point_index: int,
        splitting_point: float,
    ) -> Dict[str("left") : List, str("right") : List]:
        """
        Split dataset into left and right datasets.

        Args:
            X_features (np.array): X features from dataset
            y_features (np.array): y features from dataset
            splitting_point_index (int): index of the splitting point
            splitting_point (float): value of the splitting point

        Returns:
            Dict[str('left'):List, str('right'):List]: dictionary with left and right datasets
        """
        raise NotImplementedError

    def create_node(self, data: list) -> Node():
        """
        Create a new node

        Args:
            data (np.array): X and y features for the left or right child of the node

        Returns:
            Node(): new node
        """
        raise NotImplementedError
