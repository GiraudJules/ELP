# Standard library imports


# Third party imports
import numpy as np
from base_tree import BaseTree

class ClassificationTree(BaseTree):
    def __init__(self) -> None:
        super().__init__(self)

    def gini_index(self, y):
        size = len(y)
        instances = np.zeros(len(self.classes))

        for row in y:
            class_label = int(row[-1])
            instances[class_label] += 1

        if size > 0:
            squared_counts = [(val/size)**2 for val in instances]
            gini = 1 - np.sum(squared_counts)
        else:
            gini = 1

        return gini

    def calculate_gini_index(self, child):
        group_size = [len(child['left']), len(child['right'])]
        left_gini = self.gini_index(child['left'])
        right_gini =  self.gini_index(child['right'])

        weight_left = group_size[0] / np.sum(group_size)
        weight_right = group_size[1] / np.sum(group_size)
        weighted_gini = weight_left * left_gini + weight_right * right_gini

        return weighted_gini
