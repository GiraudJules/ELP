#####################
#This Python file contains helper functions to be used to visualize the predictions and performances of trained models. 
######################

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pandas as pd

#####################
#Version of the plots using graph_objects
#####################

#SCATTER PLOT FUNCTION
def scatter_plot_graph(model_name, y_test, y_pred):
    '''
    The purpose of the function is to make a scatter plot to compare the ground truth y values and the predicted y values for a given model.
    This function uses graph_objects (may not be visible on some notebooks)
    Input:
        - model_name: The name of the model we are analysing 
        - y_test: The ground truth output values (y_test)
        - y_pred: The predicted output values made by the model (y_pred)
    '''
    #creating a Scatter object to make the scatter plot
    trace = go.Scatter(
        x = y_test, y = y_pred, mode = 'markers',
        opacity = 0.5, marker = dict(size = 8, color = 'blue'),
        name = 'Predicted vs Ground Truth'
        )
    
    #creating a Layout object to contain the plot information
    layout = go.Layout(
        title = 'Scatter plot on test set for {} model'.format(model_name),
        xaxis = dict(title = 'Ground truth y values'),
        yaxis = dict(title = 'Predicted y values')
        )

    #creating a Figure object containing the Scatter and the Layout
    fig = go.Figure(data = trace, layout = layout)

    #displaying figure
    fig.show()

#HISTOGRAM FUNCTION
def hist_plot_graph(model_name, y_test, y_pred):
    '''
    The purpose of the function is to make a histogram plot to compare the ground truth y values and the predicted y values for a given model.
    This function uses graph_objects (may not be visible on some notebooks)
    Input:
        - model_name: The name of the model we are analysing 
        - y_test: The ground truth output values (y_test)
        - y_pred: The predicted output values made by the model (y_pred)
    '''
    #computing the error for each prediction
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
def bar_plot_error_graph(model, model_name, y_test, y_pred):
        '''
    The purpose of the function is to make a bar plot to compare the ground truth y values and the predicted y values for 20 observations.
    This function uses graph_objects (may not be visible on some notebooks)
    Input:
        - model: trained model 
        - model_name: The name of the model we are analysing 
        - y_test: The ground truth output values (y_test)
        - y_pred: The predicted output values made by the model (y_pred)
    '''

    #creating a dataframe with 20 y_test VS y_pred
    error_airbnb = pd.DataFrame({
        'Ground Truth Values': np.array(y_test).flatten(),
        'Predicted Values': y_pred.flatten()}).reset_index().head(20)
    
    #creating 2 Bar objects to contain the 2 different bar types for the plots 
    bars = [go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),
            go.Bar(name='Ground Truth', x=error_airbnb.index, y=error_airbnb['Ground Truth Values'])
            ]
    
    #creating a Layout object
    layout = go.Layout(
        title='Test set: Predicted y vs Ground Truth y', 
        xaxis_title='Observations', 
        yaxis_title='Price'
        )
    
    #creating a Figure object containing the Bar objects
    fig = go.Figure(data= bars, layout=layout)

    #modifying layout
    fig.update_layout(barmode='group')

    #displaying figure
    fig.show()

    
#####################
#Version of the plots using Seaborn and matplotlib
#####################

#SCATTER PLOT FUNCTION
def scatter_plot(model_name, y_test, y_pred):
    '''
    The purpose of the function is to make a scatter plot to compare the ground truth y values and the predicted y values for a given model.
    Input:
        - model_name: The name of the model we are analysing 
        - y_test: The ground truth output values (y_test)
        - y_pred: The predicted output values made by the model (y_pred)
    '''
    # create a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    
    # add plot information
    plt.title('Scatter plot on test set for {} model'.format(model_name))
    plt.xlabel('Ground truth y values')
    plt.ylabel('Predicted y values')
    
    # display plot
    plt.show()

#HISTOGRAM FUNCTION
def hist_plot(model_name, y_test, y_pred):
    '''
    The purpose of the function is to make a histogram plot to compare the ground truth y values and the predicted y values for a given model.
    Input:
        - model_name: The name of the model we are analysing 
        - y_test: The ground truth output values (y_test)
        - y_pred: The predicted output values made by the model (y_pred)
    '''
    # calculate errors
    error = y_pred - y_test

    # create histogram plot
    plt.figure(figsize=(8, 6))
    sns.histplot(error, bins=50)

    # add plot information
    plt.title('Histogram for errors on test set for {} model'.format(model_name))
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.xlim(-1500, 1500)
    
    # display plot
    plt.show()

