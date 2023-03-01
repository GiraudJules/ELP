class Node:
    """
    Class to build a Node.
    """

    def __init__(self, value, is_leaf=False):
        self.splitting_point = value
        self.left_child = None
        self.right_child = None
        self.risk_value = None
        self.col_index = None
        self.is_leaf = is_leaf
