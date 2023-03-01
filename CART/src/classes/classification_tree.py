# Standard library imports


# Third party imports
import numpy as np


class ClassificationTree():
    def __init__(self, max_depth:int, min_sample_leaf:int, min_samples_split:int):
    def fit(self, X_features:np.array, y_features: np.array):
        pass
    def create_nodes(self, X_features:np.array, y_features: np.array):
        pass
    def get_split(self, X_features:np.array, y_features: np.array, splitting_value_index: int, splitting_value: float):
        pass
    def build_tree(self, node: Node(), current_depth: int):
        pass
    def predict(self, X_test:np.array):
        pass
