import pandas as pd
import numpy as np

def gini_index(groups, classes):
	"""
    Compute the Gini impurity

    Args:
        groups (list): List of values with the associated label
        classes (list): Target labels

    Returns:
        float: Gini impurity
    """
	n_instances = float(sum([len(group) for group in groups]))
	gini = 0
	
	for group in groups:
		size = float(len(group))
		
		if size == 0:
			continue
		score = 0.0
		
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		
		gini += (1.0 - score) * (size / n_instances)
	
	return gini