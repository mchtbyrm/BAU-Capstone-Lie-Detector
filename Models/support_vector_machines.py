from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def support_vector_machines(x_train, y_train, x_test, C, gamma, kernel):
    # create the model
    model = SVC(C=C, gamma=gamma, kernel=kernel)
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def choose_parameters(x_train, y_train):
    # create the model
    model = SVC()
    # define the parameter values that should be searched
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    # instantiate the grid. refit=True means that the best model is refitted to the entire dataset
    # verbose=3 means that the progress of the grid search is printed
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3)
    # fit the grid with data
    grid.fit(x_train, y_train)
    # print the best parameters
    print(grid.best_params_)
    # print the best estimator
    print(grid.best_estimator_)
    return grid.best_params_
