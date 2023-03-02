# Standard library imports
from abc import abstractmethod
from typing import Union, Dict, List

# Third party imports
import numpy as np

# Local applications imports
from src.classes.node import Node


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

    def fit(self, X_features: np.array, y_features: np.array) -> Node:
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
            if node.left is not None and isinstance(node.left, list):
                # Instanciate a new node from left child
                left_node = self.create_node(node.left)
                node.left = left_node
                # While left node is not a leaf, build the tree
                if node.left.is_leaf is not True:
                    self.build_tree(node.left, current_depth + 1)

            # If right child is a list
            if node.right is not None and isinstance(node.right, list):
                # Instanciate a new node from right child
                right_node = self.create_node(node.right)
                node.right = right_node
                # While left node is not a leaf, build the tree
                if node.right.is_leaf is not True:
                    self.build_tree(node.right, current_depth + 1)

    def split_dataset(
        self,
        X_features: np.array,
        y_features: np.array,
        splitting_point: float,
    ) -> Dict[str("left") : [list, list], str("right") : [list, list]]:
        """
        Split dataset into left and right datasets.

        Args:
            X_features (np.array): X values of ONE specific feature from dataset
            y_features (np.array): y of ONE specific feature from dataset
            splitting_point (float): value of the splitting point

        Returns:
            Dict[str('left'):List[list,list], str('right'):List[list,list]]: dictionary with left and right datasets
        """

        left_x_data = []
        left_y_data = []

        right_x_data = []
        right_y_data = []

        for i, val in enumerate(X_features):
            if y_features[i] <= splitting_point:
                left_x_data.append(val)
                left_y_data.append(y_features[i])

            if y_features[i] > splitting_point:
                right_x_data.append(val)
                right_y_data.append(y_features[i])

        left_child = [left_x_data, left_y_data]
        right_child = [right_x_data, right_y_data]

        return {"left": left_child, "right": right_child}

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
