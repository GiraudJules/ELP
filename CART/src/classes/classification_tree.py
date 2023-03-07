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
            data (_type_): The target values of the dataset.

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
    
    def most_common_label(self, data):
        """
        Returns the most common class label in the given dataset.

        Args:
            data (list): List of lists, where each inner list represents a data point, and the last element of each inner
                list represents the class label.

        Returns:
            int: The most common class label in the given dataset.
        """
        labels = [row[-1] for row in data]
        return max(set(labels), key=labels.count)
    
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
        
        gini_index = 1.0
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
                        node.predicted_value = self.most_common_label(child)
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
        print(f"-----------------------------> Risk value: {gini_index}")
        print(f"-----------------------------> Predicted value: {node.predicted_value}")
        print(f"-----------------------------> Is leaf: {node.is_leaf}")
        print("=" * 50 + "End of node creation" + "=" * 50)

        return node


    # def create_node(self, data) -> Node:
    #     """Create a new node of the classification tree.

    #     Args:
    #         data (numpy.ndarray): The dataset to split.

    #     Returns:
    #         Node: The created node.
    #     """

    #     print("=" * 50 + "Creating new node..." + "=" * 50)
    #     node = Node(None)
    #     print(f"Lenght of data: {len(data)}")
        
    #     gini = self.gini_index(data)

    #     # Check if node has enough samples to be split again
    #     if len(data) <= self.min_sample_split:
    #         node.is_leaf = True
    #         target_values = [row[-1] for row in data]
    #         node.value = np.bincount(target_values).argmax()
    #         node.gini_value = gini
    #         return node

    #     gini_index = 1.0

    #     for col_index in range(len(data[0])-1):
    #         for row_index in range(len(data)):
    #             value = data[row_index][col_index]
    #             child = self.split_dataset(data, col_index, value)

    #             # Check the minimum number of samples required to be at a leaf node
    #             if (len(child['left']) >= self.min_sample_leaf & len(child['right']) >= self.min_sample_leaf):
    #                 node_gini_index = self.calculate_gini_index(child)

    #                 if node_gini_index < gini_index:
    #                     gini_index = node_gini_index
    #                     node.value = value
    #                     node.gini_value = node_gini_index
    #                     node.column_index = col_index
    #                     node.left_region = child['left']
    #                     node.right_region = child['right']

    #     return node
