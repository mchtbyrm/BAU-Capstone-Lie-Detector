from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB


def gaussian_naive_bayes(x_train, y_train, x_test):
    # create the model
    model = GaussianNB()
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return predictions
