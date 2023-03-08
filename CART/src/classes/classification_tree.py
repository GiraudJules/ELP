# Standard library imports


# Third party imports
import numpy as np
import pandas as pd
from src.classes.base_tree import BaseTree
from src.classes.node import Node

class ClassificationTree(BaseTree):
    def __init__(self, min_sample_leaf, max_depth, min_sample_split) -> None:
        super().__init__(min_sample_leaf, max_depth, min_sample_split)

    def gini_index(self, data: pd.DataFrame) -> np.float64:
        """Calculate the Gini index for a given dataset.

        Args:
            data (pd.DataFrame): Dataframe representing the child samples

        Returns:
            np.float64: The Gini index for the given dataset.
        """
        size = len(data)
        instances = np.array(data.groupby(data.columns[-1]).size())
        
        gini_index = 1.0 - np.sum((instances / size) ** 2)
        return gini_index

    def weighted_gini_index(self, child: dict) -> np.float64:
        """Calculate the weighted Gini index for a given split.

        Args:
            child (dict): A dictionary containing the left and right child nodes resulting from a split.

        Returns:
            np.float64: The weighted Gini index for the given split.
        """
        group_size = [len(child['left']), len(child['right'])]

        left_gini = self.gini_index(child['left'])
        right_gini =  self.gini_index(child['right'])
        
        weight_left = group_size[0] / np.sum(group_size)
        weight_right = group_size[1] / np.sum(group_size)

        weighted_gini = weight_left * left_gini + weight_right * right_gini

        return weighted_gini
    
    def majority_vote(self, data: pd.DataFrame) -> int:
        """Return the class index (y) of the class which have the most samples in a node

        Args:
            data (pd.DataFrame): Dataframe representing the node samples

        Returns:
            int: Class index
        """
        instances = [0] * len(self.classes)
        target = data.iloc[:, -1].astype(int)

        for class_label in target:
            instances[class_label] += 1

        y_pred = instances.index(max(instances))

        return y_pred
    
    def create_node(self, data) -> Node:
        """
        Create a new node

        Args:
            data (list): List of X and y features for the left or right child of the node

        Returns:
            Node: New node
        """
        node = Node(None)

        # Check if node has enough samples to be split again
        if len(data) <= self.min_sample_split:
            node.is_leaf = True
            node.predicted_value = self.majority_vote(data)
            return node
        
        gini_index = 1
        
        for col_index in range(len(data.columns) - 1):
        
            data_sorted = data.sort_values(by=data.columns[col_index])
            for row_index in range(len(data_sorted) - 1):
        
                splitting_value = (
                    data_sorted.iloc[row_index][col_index]
                    + data_sorted.iloc[row_index + 1][col_index]
                ) / 2
        
                child = self.split_dataset(data_sorted, col_index, splitting_value)
        
                # Check the minimum number of samples required to be at a leaf node
                if (len(child["left"]) >= self.min_sample_leaf) and (
                    len(child["right"]) >= self.min_sample_leaf
                ):
                  
                    candidate_gini_index = self.weighted_gini_index(child)
                   
                    # If it is current lowest Gini index, we update the node
                    if candidate_gini_index < gini_index:
                   
                        gini_index = candidate_gini_index

                        ## Update the node
                        # Which value to separate data
                        node.splitting_point = splitting_value
                       
                        # Index of the feature X
                        node.column_index = col_index
                       
                        # Set of predicted value for this node
                        node.predicted_value = self.majority_vote(data)
                       
                        # Set of X/y which go to left and right
                        node.left_region = child["left"]
                        node.right_region = child["right"]
                       
                        # We set that the node is not a leaf
                        node.is_leaf = False
                       
        return node
