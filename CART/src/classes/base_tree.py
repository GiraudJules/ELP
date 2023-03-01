# Standard library imports
from abc import ABC, abstractmethod
from typing import Union

# Third party imports
import numpy as np

#Local applications imports

class BaseTree(ABC):
    """
    Base class to build a Classification or Regression Tree.
    To build a child of this class and inherit the methods, need to implement the methods.
    """

    def __init__(self,min_sample_leaf,max_depth,min_sample_split):
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.classes = None

    @abstractmethod
    def fit(self, X_features:np.array, y_features: np.array) -> self.root:
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
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test:Union[Union[int,str],np.array]) -> Union[Union[int, str], np.array]:
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
        ""
    
        raise NotImplementedError
        
    @abstractmethod
    def split_dataset(self,)
        
    @abstractmethod
    def 