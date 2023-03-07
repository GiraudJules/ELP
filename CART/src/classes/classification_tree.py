# Standard library imports


# Third party imports
import numpy as np
import pandas as pd
from src.classes.base_tree import BaseTree
from src.classes.node import Node

class ClassificationTree(BaseTree):
    def __init__(self, min_sample_leaf, max_depth, min_sample_split) -> None:
        super().__init__(min_sample_leaf, max_depth, min_sample_split)
        self.lendf = None

    def gini_index(self, data):
        """Calculate the Gini index for a given dataset.

        Args:
            data (_type_): The target values of the dataset.

        Returns:
            float: The Gini index for the given dataset.
        """
        size = len(data)
        if size == 0:
            return 0
        
        instances = np.zeros(len(self.classes))
        target = data.iloc[:, -1].astype(int)

        for class_label in target:
            instances[class_label] += 1

        gini_index = 1.0 - np.sum((instances / size) ** 2)

        return gini_index

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
    
    def count_labels(self, data):
        """
        Returns a dictionary containing the count of each class label in the given dataset.

        Args:
            data (list): List of lists, where each inner list represents a data point, and the last element of each inner
                list represents the class label.

        Returns:
            dict: A dictionary containing the count of each class label in the given dataset.
        """
        instances = [0] * len(self.classes)
        target = data.iloc[:, -1].astype(int)

        for class_label in target:
            instances[class_label] += 1

        return instances
    
    def create_node(self, data) -> Node:
        """
        Create a new node

        Args:
            data (list): List of X and y features for the left or right child of the node

        Returns:
            Node(): new node
        """
        print("=" * 50 + "Creating new node..." + "=" * 50)
        node = Node(None)
        print(f"Lenght of data: {len(data)}")

        # Check if node has enough samples to be split again
        if len(data) <= self.min_sample_split:
            print("Stopping Criterion : Min Sample Split")
            node.is_leaf = True
            node.predicted_value = data[data.columns[-1]].mean()
            return node
        
        gini_index = 1

        print(" Browsing all features ...")

        for col_index in range(len(data.columns) - 2):
            print("*" * 30 + f" Feature n°: {col_index} " + "*" * 30)
            data_sorted = data.sort_values(by=data.columns[col_index])
            for row_index in range(len(data_sorted) - 1):
                print("*" * 20 + f" Row index: {row_index} " + "*" * 20)
                splitting_value = (
                    data_sorted.iloc[row_index][col_index]
                    + data_sorted.iloc[row_index + 1][col_index]
                ) / 2
                print(f" ---> Splitting point: {splitting_value}")
                child = self.split_dataset(data_sorted, col_index, splitting_value)
                print(f"- len_left_child: {len(child['left'])}")
                print(f"- len_right_child: {len(child['right'])}")

                # Check the minimum number of samples required to be at a leaf node
                if (len(child['left']) >= self.min_sample_leaf & len(child['right']) >= self.min_sample_leaf):
                    print(" ======> Enough samples to split")
                    node_gini_index = self.calculate_gini_index(child)
                    print(f"- node_gini_index: {node_gini_index}")
                    # If it is current lowest Gini index, we update the node
                    if node_gini_index < gini_index:
                        print(" /!\ BEST Gini index Value ====> Updating node")
                        gini_index = node_gini_index

                        ## Update the node
                        # Which value to separate data
                        node.splitting_point = splitting_value
                        # Index of the feature X
                        node.column_index = col_index
                        # Set of predicted value for this node
                        #node.predicted_value = self.most_common_label(child)
                        node.predicted_value = self.count_labels(data)
                        # Set of X/y which go to left and right
                        node.left_region = child["left"]
                        node.right_region = child["right"]
                        # We set that the node is not a leaf
                        node.is_leaf = False
                        print(
                            "Corresponding predicted value is :", node.predicted_value
                        )
        print(
            f"-----------------------------> Node selected for feature n° {node.column_index}: {node.splitting_point}"
        )
        print(
            f"-----------------------------> Length left child: {len(node.left_region)}"
        )
        print(
            f"-----------------------------> Length right child: {len(node.right_region)}"
        )
        print(f"-----------------------------> Gini index : {gini_index}")
        print(f"-----------------------------> Predicted value: {node.predicted_value}")
        print(f"-----------------------------> Is leaf: {node.is_leaf}")
        print("=" * 50 + "End of node creation" + "=" * 50)

        return node
