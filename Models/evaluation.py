from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(y_test, predictions):
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Accuracy Score:")
    print(accuracy_score(y_test, predictions))
