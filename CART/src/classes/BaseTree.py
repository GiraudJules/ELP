# Standard library imports
from abc import ABC, abstractmethod
from typing import Union

# Third party imports
import numpy as np


class BaseTree(ABC):
    """
    Base class to build a Classification or Regression Tree.
    To build a child of this class and inherit the methods, need to implement the methods.
    """

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
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs) -> Union[int, str]:
        """Predict class for a single sample.

        Raises
        ------
        NotImplementedError: if the method is not implement
        """
        raise NotImplementedError
