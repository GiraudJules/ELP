import numpy as np

from typing import Union, List
from base_tree import BaseTree

class RegressionTree(BaseTree):
    def __init__(self) -> None:
        super().__init__(self)
        self.risk_function = self.risk_regression()

    def mse(self, y: Union[np.array, List]) -> float:
        """Compute Mean Square Error from the average of a given region R

        Args:
            y (Union[np.array, List]): Given region R (left or right)

        Returns:
            float: MSE
        """
        y_mean = np.mean(y)
        mse = np.square(y - y_mean).mean()
        return mse
    
    def risk_regression(self, child: dict) -> float:
        """Compute the risk function for the regression from the two regions 

        Args:
            child (dict): dict representing left and right regions

        Returns:
            float: Risk value of two regions
        """
        left_risk = self.mse(child['left'])
        right_risk =  self.mse(child['right'])
        return (left_risk + right_risk)
    
