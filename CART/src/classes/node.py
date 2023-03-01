# Standard library imports
from typing import List


class Node:
    """
    Class to build a Node.
    """

    def __init__(self, value, is_leaf=False):
        self.splitting_point: float = value
        self.left_child: List[list, list] = None
        self.right_child: List[list, list] = None
        self.risk_value: int = None
        self.is_leaf: bool = is_leaf
