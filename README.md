# Ensemble Learning Project (ELP) - MSc AI - CS 

## Credits
This project was created by Amandine Allmang, Nicolas Bourriez, Marie Bouvard, Jules Giraud, Arthur Nardone as a part of the Ensemble Learning course of the MSc AI at CentraleSupelec.

## *Project 1: AirBnB Price Prediction*
The goal of this project was to predict AirBnB prices in New York using ensemble learning methods. The code is written in Python and uses popular machine learning libraries such as Scikit-learn and XGBoost. 

### Dataset
The dataset used in this project is the [Airbnb New York City Airbnb Open Data dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), which contains detailed information on Airbnb listings in New York City, including price, location, room type, and availability.

### Installation
To run this code, you need to have Python 3 and the following Python libraries installed:

- pandas
- numpy
- matplotlib
- plotly.graph_objects
- seaborn
- scikit-learn
- xgboost

To install these libraries, you can use pip by running:

`pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly`

### Usage
You can run the code by opening the Jupyter Notebook `AirBnB_boosting_models.ipynb` and `AirBnB_bagging_models.ipynb` and executing the cells. The notebooks contain the training of the various models and visualizations of the results. To do so, the notebooks use the different helper methods present in the Python (.py) files.

The code consists of the following files:

- In the `Data_Visualization` folder, containing `Data_Analysis_Price_vs_Rest.ipynb` and `Uni_Variable_Data_Exploration.ipynb`, we explored the data and looked at the distribution of samples for each feature as well as the interaction of each feature with the price of the listing. 
- In `utils.py`, we take care of data cleaning, preprocessing and evaluation. In this step, we load the dataset, handle missing values, and perform some feature engineering to prepare the data for modeling.
- In `bagging.py` and `boosting.py`, we can acces functions that create bagging and boosting models respectively.
- `evaluation_plots.py` contains the functions that allow us to visualize our results. 
- `AirBnB_bagging_models.ipynb` and `AirBnB_boosting_models.ipynb` can be run to train your own Decision Trees, Random Forest, Extremely Randomized Trees, Gradient Boosting, XGBoost and Adaboost models. They currently contain our best results and corresponding plots.

## *Project 2: Decision Tree implementation*

### Dataset
The dataset used in this project is the [Iris](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) for classification task and [Diabetes](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) for regression task, both of the datasets come from scikit-learn standard datasets.

### Installation
To run this code, you need to have Python 3 and the following Python libraries installed:

- pandas
- numpy
- scikit-learn

To install these libraries, you can use pip by running:

`pip install pandas numpy scikit-learn`

### Usage
You can run the code by opening the Jupyter Notebook `cart_test.ipynb` and executing the cells. The notebooks allow you to test our regression and classification implementation. To do so, the notebooks use the different helper methods present in the Python (.py) files.

The code consists of the following files:

- The `CART` folder, containing `src` folder with `classes` folder inside. 
- Inside `classes` folder, you can find `base_tree.py` and `node.py`.
- Inside `classes` folder, you can find `classification_tree.py` and `regression_tree.py`
- There is also a notebook called `cart_test.ipynb`.

## Contributing
Contributions to the codebase are welcome. If you find any issues or have ideas for improvement, feel free to open an issue or pull request.

## Badges
![linting: pylint](https://github.com/GiraudJules/ELP/actions/workflows/pylint.yml/badge.svg)
