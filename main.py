import pandas as pd
from ReadSerialPort.read_serial_port import read_serial
from ReadSerialPort.read_serial_port2 import read_data
from generate_features import calculate_features
from data_utils import *
from Models.logistic_regression import logistic_regression
from Models.kNN import k_nearest_neighbors, choose_k
from Models.naive_bayes import gaussian_naive_bayes
from Models.support_vector_machines import support_vector_machines, choose_parameters
from Models.trees import decision_tree, random_forest, choose_n_estimators
from Models.evaluation import evaluate_model


cols = ["Gender", "Min_BPM", "Max_BPM", "Mean_BPM", "HRV", "Min_GSR", "Max_GSR", "Mean_GSR", "GSR_Variability", "Class"]

dataframe = read_dataset("lie.csv", cols)

# show_histograms(dataframe, cols)

# create_jointplot(dataframe, cols)

# create_pairplot(dataframe, cols)

x_train, x_test, y_train, y_test = split_data(dataframe, cols)

x_train, y_train = oversample_data(x_train, y_train)

x_test, y_test = oversample_data(x_test, y_test)

x_train = scale_data(x_train)

x_test = scale_data(x_test)

train_data = reshape_data(x_train, y_train)

test_data = reshape_data(x_test, y_test)

print(train_data.shape)
print(test_data.shape)

df_scaled = pd.DataFrame(x_train, columns=cols[:-1])
df_scaled['Class'] = y_train
print(df_scaled.head())

# model, predictions = logistic_regression(x_train, y_train, x_test)

k = choose_k(x_train, y_train, x_test, y_test)

model, predictions = k_nearest_neighbors(x_train, y_train, x_test, k)

# model, predictions = gaussian_naive_bayes(x_train, y_train, x_test)

# best_params = choose_parameters(x_train, y_train)
#
# model, predictions = support_vector_machines(x_train, y_train, x_test, best_params['C'], best_params['gamma'],
#                                              best_params['kernel'])

# model, predictions = decision_tree(x_train, y_train, x_test)

# best_n_estimators = choose_n_estimators(x_train, y_train, x_test, y_test)
#
# model, predictions = random_forest(x_train, y_train, x_test, best_n_estimators)

evaluate_model(predictions, y_test)

# read_serial()

# bpm_data, gsr_data = read_data('COM1', 60)
# print(bpm_data)
# print(gsr_data)
# features = calculate_features(bpm_data, gsr_data)
# features = features.reshape(1, -1)
# features_scaled = scale_data(features)
# print(features)
#
# print("*****************************************************************")
#
# if model.predict(features) == 1:
#     print("The subject is telling the truth")
# else:
#     print("The subject is lying")



