from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from matplotlib import pyplot as plt


def decision_tree(x_train, y_train, x_test):
    # create the model
    model = DecisionTreeClassifier()
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


def random_forest(x_train, y_train, x_test, n_estimators):
    # create the model
    model = RandomForestClassifier(n_estimators=n_estimators)  # n_estimators is the number of trees in the forest
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


# returns the best number of estimators
def choose_n_estimators(x_train, y_train, x_test, y_test):
    # Calculating error for K values between 1 and 40
    estimator_range = range(1, 101)

    # Initialize empty list to store the error rates
    error_rates = []

    # Loop over each value of n_estimators to train and test a Random Forest Classifier
    for n_estimators in estimator_range:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        rf.fit(x_train, y_train)

        # Predict on the testing data and calculate the error rate
        y_pred = rf.predict(x_test)
        error_rate = np.sum(y_pred != y_test, dtype=int) / len(y_test)

        # Store the error rate for this value of n_estimators
        error_rates.append(error_rate)

    # Find the index of the lowest error rate
    best_index = error_rates.index(min(error_rates))

    plt.figure(figsize=(10, 6))  # figsize=(a, b) a is width, b is height
    plt.plot(range(1, 101), error_rates, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate vs N Estimators Value')
    plt.xlabel('N Estimators')
    plt.ylabel('Error')
    plt.show()

    print(f"Best number of estimators: {estimator_range[best_index]}")

    # Return the corresponding value of n_estimators
    return estimator_range[best_index]

