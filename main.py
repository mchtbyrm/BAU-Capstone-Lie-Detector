from ReadSerialPort.read_serial_port import read_serial
from data_utils import *


cols = ["Gender", "Min_BPM", "Max_BPM", "Mean_BPM", "HRV", "Min_GSR", "Max_GSR", "Mean_GSR", "GSR_Variability", "Class"]

dataframe = read_dataset("lie.csv", cols)

# show_histograms(dataframe, cols)

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

# read_serial()


