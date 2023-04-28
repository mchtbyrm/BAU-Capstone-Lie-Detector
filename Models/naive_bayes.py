from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB


def gaussian_naive_bayes(x_train, y_train, x_test):
    # get the best hyperparameters
    params = grid_search_for_naive_bayes(x_train, y_train)
    # create the model
    model = GaussianNB(var_smoothing=params['var_smoothing'], priors=params['priors'])
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def grid_search_for_naive_bayes(x_train, y_train):
    # create the model
    model = GaussianNB()

    # define the hyperparameter grid
    # var_smoothing means the portion of the largest variance of all features
    # that is added to variances for calculation stability
    # priors means the prior probabilities of the classes
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
                  'priors': [None, [0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]}

    # create GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

    # fit the GridSearchCV object to the training data
    grid_search.fit(x_train, y_train)

    # print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # return the best model and its predictions
    return grid_search.best_params_
