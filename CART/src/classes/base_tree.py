# Standard library imports
from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import Union
=======
from typing import Union, Dict, List
>>>>>>> origin/dev

# Third party imports
import numpy as np

<<<<<<< HEAD
=======
# Local applications imports
from node import Node

>>>>>>> origin/dev

class BaseTree(ABC):
    """
    Base class to build a Classification or Regression Tree.
    To build a child of this class and inherit the methods, need to implement the methods.
    """

<<<<<<< HEAD
    @abstractmethod
    def fit(self, X_features, y_features):
        """
        Method to fit the build decision tree classifier.

        Parameters
        ----------
        X_features (np.array)
        y_features (np.array)

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the method is not implement
=======
    def __init__(self, min_sample_leaf, max_depth, min_sample_split):
        super().__init__()
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.classes = None
        self.risk_function = None

    @abstractmethod
    def fit(self, X_features: np.array, y_features: np.array) -> Node():
        """
        - Retrieves the different classes from X_features and stores it into self.classes
        - Assign new value to self.node with self.create_node
        - Build the tree from new root and current_depth

        Args:
            X_features (np.array): X features from dataset
            y_features (np.array): y features from dataset

        Raises:
            NotImplementedError: if the method is not implement

        Returns:
            self.root: new Node of the tree
>>>>>>> origin/dev
        """
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD
    def predict(self, inputs) -> Union[int, str]:
        """Predict class for a single sample.

        Raises
        ------
        NotImplementedError: if the method is not implement
=======
    def predict(
        self, X_test: Union[Union[int, str], np.array]
    ) -> Union[Union[int, str], np.array]:
        """
        Predict class for whether:
        - Regression: a single int OR multiple int
        - Classification: a single str OR multiple str

        Args:
            X_test (Union[Union[int,str],np.array]): test features to predict on

        Raises:
            NotImplementedError: if the method is not implement

        Returns:
            Union[Union[int, str], np.array]: whether a single int or str; or np.array

        """

        raise NotImplementedError

    @abstractmethod
    def build_tree(self, node: Node(), current_depth: int) -> None:
        """
        Build the tree recursively.

        Args:
            node (Node()): current node
            current_depth (int): current depth of the tree

        Raises:
            NotImplementedError: if the method is not implement
        """
        raise NotImplementedError

    @abstractmethod
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

        Raises:
            NotImplementedError: if the method is not implement

        Returns:
            Dict[str('left'):List, str('right'):List]: dictionary with left and right datasets
        """
        raise NotImplementedError

    @abstractmethod
    def create_node(self, X_features: np.array, y_features: np.array) -> Node():
        """
        Create a new node

        Args:
            X_features (np.array): X features from dataset
            y_features (np.array): y features from dataset

        Raises:
            NotImplementedError: if the method is not implement

        Returns:
            Node(): new node
>>>>>>> origin/dev
        """
        raise NotImplementedError
