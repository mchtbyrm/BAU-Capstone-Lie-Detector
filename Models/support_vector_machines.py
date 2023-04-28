from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def support_vector_machines(x_train, y_train, x_test):
    # find the best hyperparameters
    params = grid_search_for_svm(x_train, y_train)
    # create the model
    model = SVC(C=params['C'], kernel=params['kernel'])
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def grid_search_for_svm(x_train, y_train):
    # create the model
    model = SVC()

    # define the hyperparameter grid
    param_grid = {'C': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    # create GridSearchCV object
    # cv: Number of cross - validation splits to use for evaluating the performance of each hyperparameter combination.
    # n_jobs: Number of CPU cores to use for parallelize the grid search process.
    # If set to -1, all available cores will be used.
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

    # fit the GridSearchCV object to the training data
    grid_search.fit(x_train, y_train)

    # print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # return the best model and its predictions
    return grid_search.best_params_
