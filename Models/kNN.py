import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# k is the number of neighbors
def k_nearest_neighbors(x_train, y_train, x_test, k):
    # create the model
    model = KNeighborsClassifier(n_neighbors=k)
    # train the model
    model.fit(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    return model, predictions


# choose the best k value
def choose_k(x_train, y_train, x_test, y_test):
    error = []
    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        model = KNeighborsClassifier(n_neighbors=i)
        model = model.fit(x_train, y_train)
        pred_i = model.predict(x_test)
        error.append(np.mean(pred_i != y_test))
    # Find the index of the lowest error rate
    best_index = error.index(min(error))
    plt.figure(figsize=(10, 6))  # figsize=(a, b) a is width, b is height
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate vs K Value')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.show()

    print(f"Best k value: {best_index + 1}")

    return best_index + 1
