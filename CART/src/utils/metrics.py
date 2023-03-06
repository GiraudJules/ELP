from src.classes import regression_tree, classification_tree

def accuracy(train_data, test_data, type: str):
    if type == 'regression':
        dt = regression_tree.RegressionTree()
        dt = dt.fit(train_data)


    # predictions = dt.predict(data[0:-1][0:-1])
    # true_values = [row[-1] for row in data]
    # return '{:1f}'.format(sum([t==p for t,p in zip(true_values, predictions)])/len(true_values) *100 ) + '% accuracy' 