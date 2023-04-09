from sklearn.linear_model import LogisticRegression


def logistic_regression(x_train, y_train, x_test):
    # create the model
    model = LogisticRegression()
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return predictions
