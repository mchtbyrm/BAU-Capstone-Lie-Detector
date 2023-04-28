from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from matplotlib import pyplot as plt


def decision_tree(x_train, y_train, x_test):
    # find the best hyperparameters
    params = grid_search_for_decision_tree(x_train, y_train)
    # create the model
    model = DecisionTreeClassifier(criterion=params['criterion'],max_depth=params['max_depth'],
                                   min_samples_split=params['min_samples_split'],
                                   min_samples_leaf=params['min_samples_leaf'])
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def grid_search_for_decision_tree(x_train, y_train):
    # create the model
    model = DecisionTreeClassifier()

    # define the hyperparameter grid
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    # create GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

    # fit the GridSearchCV object to the training data
    grid_search.fit(x_train, y_train)

    # print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # return the best model and its predictions
    return grid_search.best_params_


def random_forest(x_train, y_train, x_test):
    # find the best hyperparameters
    params = grid_search_for_random_forest(x_train, y_train)
    # create the model
    model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'],
                                   max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
                                   min_samples_leaf=params['min_samples_leaf'])
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def grid_search_for_random_forest(x_train, y_train):
    # create the model
    model = RandomForestClassifier()

    # define the hyperparameter grid
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [2, 4, 6, 8, 10, 12, 14, 16],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    # create GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

    # fit the GridSearchCV object to the training data
    grid_search.fit(x_train, y_train)

    # print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # return the best model and its predictions
    return grid_search.best_params_

