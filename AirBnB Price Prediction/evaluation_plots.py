#####################
#This Python file contains helper functions to be used to visualize the predictions and performances of trained models.
######################

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pandas as pd

#SCATTER PLOT FUNCTION
def scatter_plot(model_name, y_test, y_pred):
    '''
    The purpose of the function is to make a scatter plot to compare the ground truth y values and the predicted y values for a given model.
    Input:
        - The name of the model we are analysing 
        - The ground truth output values (y_test)
        - The predicted output values made by the model (y_pred)
    '''
    #creating a Scatter object to make the scatter plot
    trace = go.Scatter(
        x = y_test, y = y_pred, mode = 'markers',
        opacity = 0.5, marker = dict(size = 8, color = 'blue'),
        name = 'Predicted vs Ground Truth'
        )
    
    #creating a Layout object to contain the plot
    layout = go.Layout(
        title = 'Scatter plot on test set for {} model'.format(model_name),
        xaxis = dict(title = 'Ground truth y values'),
        yaxis = dict(title = 'Predicted y values')
        )

    #creating a Figure object contianing the Scatter and the Layout
    fig = go.Figure(data = trace, layout = layout)

    #displaying figure
    fig.show()

#HISTOGRAM FUNCTION
def hist_plot(model_name, y_test, y_pred):
    '''
    The purpose of the function is to make a histogram plot to compare the ground truth y values and the predicted y values for a given model.
    Input:
        - The name of the model we are analysing 
        - The ground truth output values (y_test)
        - The predicted output values made by the model (y_pred)
    '''
    #compuuting the error for each prediction
    error = (y_pred - y_test)

    #creating a Histogram object
    data = go.Histogram(x=error, nbinsx=50)

    #creating a Layout object
    layout = go.Layout(
        title='Histogram for errors on test set for {} model'.format(model_name), 
        xaxis=dict(title='Error', range=[-1500, 1500]), 
        yaxis_title='Frequency'
        )
    
    #creating a Figure object containing the Histogram and the Layout
    fig = go.Figure(data=data, layout=layout)

    #displaying figure
    fig.show()

#BAR PLOT FOR ERRORS FUNCTION
def bar_plot_error(model, model_name, y_test, y_pred):

    #creating a dataframe with 20 y_test VS y_pred
    error_airbnb = pd.DataFrame({
        'Ground Truth Values': np.array(y_test).flatten(),
        'Predicted Values': y_pred.flatten()}).reset_index().head(20)
    
    #creating 2 Bar objects to contain the 2 different bar types for the plots 
    bars = [go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),
            go.Bar(name='Ground Truth', x=error_airbnb.index, y=error_airbnb['Ground Truth Values'])
            ]
    
    #creating a Figure object containing the Bar objects
    fig = go.Figure(data= bars)

    #modifying layout and title
    fig.update_layout(barmode='group')
    title = 'Test set: Predicted y vs Ground Truth y'
    fig.update_layout(title=title)

    #displaying figure
    fig.show()

#FEATURE IMPORTANCE PLOT FUNCTION
def plot_feature_importance(model, feature_names):
    '''
    This function allows to plot the feature importance of a given model
    Input:
        - An initialized model
        - A list of feature names
    '''
    #getting the feature importances from the model
    importances = model.feature_importances_

    #sorting feature importances in descending order
    indices = importances.argsort()[::-1]

    # matching feature names with feature importance order 
    sorted_names = [feature_names[i] for i in indices]

    #horizontal bar chart of feature importances
    plt.barh(range(len(indices)), importances[indices], align='center')

    #y-axis ticks to be feature names
    plt.yticks(range(len(indices)), sorted_names)

    #add labels
    plt.xlabel('Relative Importance')
    plt.title('Feature Importances')

    #plot
    plt.show()