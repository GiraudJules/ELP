import numpy as np

from base_tree import BaseTree

class RegressionTree(BaseTree):
    def __init__(self) -> None:
        super().__init__()
        self.risk_function = None

    def mse(self, y: np.array):
        y_mean = np.mean(y)
        mse = np.square(y - y_mean).mean()
        return mse
    
    def risk_regression(self, value, child):
        
    
