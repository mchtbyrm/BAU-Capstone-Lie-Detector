import os
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# k is the number of neighbors
def k_nearest_neighbors(x_train=None, y_train=None, x_test=None):
    model_filename = 'knn_model_lie_detector.joblib'

    # Check if the model file already exists
    if os.path.exists(model_filename):
        # Load the existing model from disk
        model = joblib.load(model_filename)
        predictions = None
        print("Model loaded from disk")
    else:
        # Model file doesn't exist, train the model
        params = choose_k_with_gridsearch(x_train, y_train)
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        # model = KNeighborsClassifier(n_neighbors=1)
        model.fit(x_train, y_train)

        # Save the trained model to disk
        joblib.dump(model, model_filename)

        # Make predictions
        predictions = model.predict(x_test)
    return model, predictions


def choose_k_with_gridsearch(x_train, y_train):
    # Define the KNN model
    knn = KNeighborsClassifier()

    # Define the parameters to search over
    param_grid = {'n_neighbors': np.arange(1, 40)}

    # Create the GridSearchCV object
    knn_cv = GridSearchCV(knn, param_grid, cv=5)  # cv is the number of folds. so 5 means 5-fold cross validation

    # Fit the GridSearchCV object to the data
    knn_cv.fit(x_train, y_train)

    plt.figure()
    plt.plot(knn_cv.cv_results_['param_n_neighbors'], knn_cv.cv_results_['mean_test_score'], color='red',
             linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.xlabel('n_neighbors')
    plt.ylabel('Mean accuracy')
    plt.title('Grid Search Results')
    plt.show()

    # Print the best parameter and its accuracy score
    print(f"Tuned Hyperparameters: {knn_cv.best_params_}")
    print("Best Accuracy Score: {:.2f}%".format(knn_cv.best_score_ * 100))

    # Return the best model and its predictions
    return knn_cv.best_params_

