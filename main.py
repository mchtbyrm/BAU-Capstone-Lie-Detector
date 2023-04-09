from ReadSerialPort.read_serial_port import read_serial
from data_utils import *
from Models.logistic_regression import logistic_regression
from Models.kNN import k_nearest_neighbors, choose_k
from Models.naive_bayes import gaussian_naive_bayes
from Models.support_vector_machines import support_vector_machines
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

# predictions = logistic_regression(x_train, y_train, x_test)

# choose_k(x_train, y_train, x_test, y_test)
#
# predictions = k_nearest_neighbors(x_train, y_train, x_test, 1)

# predictions = gaussian_naive_bayes(x_train, y_train, x_test)

predictions = support_vector_machines(x_train, y_train, x_test)

evaluate_model(predictions, y_test)

# read_serial()


