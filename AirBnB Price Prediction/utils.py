from sklearn.model_selection import train_test_split, KFold, cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV


def preprocessing(df, to_drop = ['id','host_name','last_review', 'name', 'host_id'], test_size=0.2):
    '''
    The purpose of this function is to preprocess our data. We drop columns that could induce suprious correlations 
    between the data and output, we split numerical and categorical features and perform one-hot-encoding on 
    categorical features. We remove the price from numercial features and save it as our target, y. We then concatenate 
    the numercial and categorical features to obtain data that can be used for training and testing. 
    We split our data into train and test sets with associated labels, and finally we scale them.

    Input:
        - A dataframe containing the data
        - A list of columns to drop. If not specified, the columns 'id','host_name','last_review', 'name', 'host_id' will be dropped.
        - A number between 0 and 1 representing the size of the test set. If not specified, it will be 0.2. 

    Output:
        - Training data and the associated labels
        - Test data and the associated labels
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

    # Splitting our data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Normalizing our data with a RobustScaler, which is resistant to outliers
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test



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
        
        


def eval(model, X_train, y_train, n_folds=5, scores = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']):
    '''
    The purpose of the function is to evaluate and display the performance of our model on the train and test sets. 
    Input:
        - An initialized model
        - Training data and the associated labels
        - The numerical features of the data, excluding the variable we are looking to predict
        - The number of folds for the K-Fold cross validation. If no number is specified, the default is 5.
        - A list of metrics we want to looks at. If none are specified by the user, these will be 
          the R2 score, the mean absolute error and the root mean squared error. 

    Output:
        - A dictionnary containing the training and test score for all metrics given as input on the given model.
    '''

    # Initialize the K-Fold cross-validation. By default, there will be 5 folds.
    kf = KFold(n_folds, shuffle=True, random_state = 91)

    # Cross-validate the model over 5 folds. 
    sc = cross_validate(model, X_train, y_train, scoring=scores, return_train_score=True, cv=kf)

    # Print the scores
    print('Average run time for {} folds on a {}: {:.5f} +/- {:5f}'.format(n_folds, model, abs(sc['fit_time'].mean()), abs(sc['fit_time'].std())))
    print('-'*60)

    print_score_dict(sc)
    
    return sc




