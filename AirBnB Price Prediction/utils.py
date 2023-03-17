'''
A python file containing helpers functions for data preprocessing and model evaluation.
Author: Marie Bouvard
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_validate


def preprocessing(df, to_drop = ['id','host_name','last_review', 'name', 'host_id']):
    '''
    The purpose of this function is to preprocess our data. We drop columns that could induce suprious correlations 
    between the data and output, we split numerical and categorical features and perform one-hot-encoding on 
    categorical features. We remove the price from numercial features and save it as our target, y. We then concatenate 
    the numercial and categorical features to obtain data that can be used for training and testing. 
    We split our data into train and test sets with associated labels, and finally we scale them.

    Input:
        - df: a dataframe containing the data
        - to_drop: a list of columns to drop. If not specified, the columns 'id','host_name','last_review', 'name', 'host_id' will be dropped. 

    Output:
        - X, y: Data and the associated labels
    '''
    # Dropping unused features and filling 
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df.drop(to_drop, axis=1, inplace=True)

    # OHE of categorical features
    categorical_features = df.select_dtypes(include=['object'])
    categorical_features_one_hot = pd.get_dummies(categorical_features, drop_first=True)

    # Separating price form other numerical features
    numerical_features =  df.select_dtypes(exclude=['object'])
    y = numerical_features.price
    numerical_features = numerical_features.drop(['price'], axis=1)

    # Creating our X variables
    X = np.concatenate((numerical_features, categorical_features_one_hot), axis=1)

    return X, y
    

def splits(X,y, test_size=0.2, y_preproc=False):
    '''
    This function takes data and the associated labels and splits it to create a training and a test set. 

    Inputs:
        - X: the data
        - y: the associated labels
        - test_size: a float between 0 and 1 representing the size of the test set. If not specified, it will be 0.2.
        - y_preproc: a boolean for whether or not to preprocess the y values
    Outpust:
        - X_train, y_train: training data and the associated labels
        - X_test, y_test: test data and the associated labels
    '''
    # Splitting our data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalizing our data with a RobustScaler, which is resistant to outliers
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    #if required: normalize y 
    if y_preproc==True:
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.fit_transform(y_test)

    return X_train, X_test, y_train, y_test

def search_parameters(model, parameters, X_train, y_train):
    '''
    The purpose of this function is to find optimal paramters for a bagging model using GridSearchCV.

    Input:
        - model: a bagging model
        - parameters: a dictionnary of all the parameters we want to investigate
        - X_train: training data
        - y_train: the labels associated to the data
    Output:
        - best_params: the best parameters found by GridSearchCV. 
    '''
    grid = GridSearchCV(model, parameters)
    grid.fit(X_train,y_train)
    best_params = grid.best_params_
    return best_params

def print_score(scores, sc):
    '''
    A helper function to print the scores from our evaluation function from a list of metrics.
    '''
    for s in scores:
        train_name = 'train_' + s
        test_name = 'test_' + s
        name = s.split('_')
        if name[0] == 'neg': ' '.join(name[1:])
        else: ''.join(name)

        print('Train '+ name+ ': {:.5f} +/- {:5f}'.format(abs(sc[train_name].mean()), abs(sc[train_name].std())))
        print('Test '+ name  +': {:.5f} +/- {:5f}'.format(abs(sc[test_name].mean()), abs(sc[test_name].std())))
        print('-'*60)


def print_score_dict(score_dict):
    '''
    A helper function to print the scores from our evaluation function from a list of metrics.
    This version uses dictionaries.
    '''
    line = 0
    for metric, scores in score_dict.items():
        name = metric.split('_')
        if name[1] == 'neg': 
            name = name[:1] + name[2:]
            name = ' '.join(name)
        else: name = ' '.join(name)
        
        print(name.title()+ ': {:.5f} +/- {:5f}'.format(abs(scores.mean()), abs(scores.std())))
        line+=1
        if line % 2 == 0 and line != 0:
            print('-'*60)
     


def eval(model, X, y, n_folds=5, scores = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']):
    '''
    The purpose of the function is to evaluate and display the performance of our model on the train and test sets. 
    Input:
        - model: An initialized model
        - X, y: Training data and the associated labels
        - n_folds: The number of folds for the K-Fold cross validation. If no number is specified, the default is 5.
        - scores: A list of metrics we want to looks at. If none are specified by the user, these will be 
          the R2 score, the mean absolute error and the root mean squared error. 

    Output:
        - sc: A dictionnary containing the training and test score for all metrics given as input on the given model.
    '''

    # Initialize the K-Fold cross-validation. By default, there will be 5 folds.
    kf = KFold(n_folds, shuffle=True, random_state = 91)

    # Cross-validate the model over 5 folds. 
    sc = cross_validate(model, X, y, scoring=scores, return_train_score=True, cv=kf)

    # Print the scores
    print('Average run time for {} folds on a {}: {:.5f} +/- {:5f}'.format(n_folds, model, abs(sc['fit_time'].mean()), abs(sc['fit_time'].std())))
    print('-'*60)

    print_score_dict(sc)
    
    return sc




