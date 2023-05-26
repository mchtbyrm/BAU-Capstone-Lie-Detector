from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def logistic_regression(x_train, y_train, x_test):
    # call the parameter_tuning function to get the best hyperparameters
    params = parameter_tuning(x_train, y_train)
    # create the model
    model = LogisticRegression(penalty=params['penalty'], C=params['C'], solver=params['solver'], max_iter=1000)
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def parameter_tuning(x_train, y_train):
    # create the model
    model = LogisticRegression()

    # define the hyperparameter grid
    param_grid = {'penalty': ['l1', 'l2', 'elasticnet'],
                  'C': [0.001, 0.01, 0.1, 1, 10],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

    # create GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

    # fit the GridSearchCV object to the training data
    grid_search.fit(x_train, y_train)

    # print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # return the best model and its predictions
    return grid_search.best_params_
