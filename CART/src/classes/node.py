# Standard library imports
from typing import List


class Node:
    """
    Class to build a Node.

    Attributes:
        splitting_point (float): value to split the data
        left_child (List[list, list]): left child of the node
        right_child (List[list, list]): right child of the node
        column_index (int): index of the column to split
        is_leaf (bool): whether the node is a leaf or not

    Methods:
        None
    """

    def __init__(self, value, is_leaf=False):
        self.splitting_point: float = value
        self.left_child: List[list, list] = None
        self.right_child: List[list, list] = None
        self.column_index: int = None
        self.is_leaf: bool = is_leaf
        self.predicted_value: float = None
