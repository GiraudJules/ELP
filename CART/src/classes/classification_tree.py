# Standard library imports


# Third party imports
import numpy as np
from src.classes.base_tree import BaseTree
from src.classes.node import Node

class ClassificationTree(BaseTree):
    def __init__(self, min_sample_leaf, max_depth, min_sample_split) -> None:
        super().__init__(min_sample_leaf, max_depth, min_sample_split)

    def gini_index(self, data):
        """Calculate the Gini index for a given dataset.

        Args:
            y (_type_): The target values of the dataset.

        Returns:
            float: The Gini index for the given dataset.
        """
        size = len(data)
        instances = np.zeros(len(self.classes))

        for row in data:
            class_label = int(row[-1])
            instances[class_label] += 1

        if size > 0:
            squared_counts = [(val/size)**2 for val in instances]
            gini = 1 - np.sum(squared_counts)
        else:
            gini = 1

        return gini

    def calculate_gini_index(self, child):
        """Calculate the weighted Gini index for a given split.

        Args:
            child (dict): A dictionary containing the left and right child nodes resulting from a split.

        Returns:
            float: The weighted Gini index for the given split.
        """
        group_size = [len(child['left']), len(child['right'])]
        left_gini = self.gini_index(child['left'])
        right_gini =  self.gini_index(child['right'])

        weight_left = group_size[0] / np.sum(group_size)
        weight_right = group_size[1] / np.sum(group_size)
        weighted_gini = weight_left * left_gini + weight_right * right_gini

        return weighted_gini

    def create_node(self, data):
        """Create a node of the decision tree.

        Args:
            data (numpy.ndarray): The dataset to split.

        Returns:
            Node: The created node.
        """
        node = Node(None)
        gini = self.gini_index(data)

        if len(data) <= self.min_sample_split:
            node.is_leaf = True
            target_values = [row[-1] for row in data]
            node.value = np.bincount(target_values).argmax()
            node.gini_value = gini
            return node

        gini_index = 1.0
        for col_index in range(len(data[0])-1):
            for row_index in range(len(data)):
                value = data[row_index][col_index]
                child = self.split_dataset(data, col_index, value)
                node_gini_index = self.calculate_gini_index(child)

                if node_gini_index < gini_index:
                    gini_index = node_gini_index
                    node.value = value
                    node.gini_value = node_gini_index
                    node.col_index = col_index
                    node.left = child['left']
                    node.right = child['right']

        return node
