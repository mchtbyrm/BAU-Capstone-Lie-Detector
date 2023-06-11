from GUI.gui import GUI
from ReadSerialPort.read_serial_port import read_serial
from data_utils import *
from Models.logistic_regression import logistic_regression
from Models.kNN import k_nearest_neighbors
from Models.naive_bayes import gaussian_naive_bayes
from Models.support_vector_machines import support_vector_machines
from Models.trees import decision_tree, random_forest
from Models.evaluation import evaluate_model

cols = ["MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR",
        "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW", "MEAN_REL_RR", "MEDIAN_REL_RR",
        "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR", "SDRR_RMSSD_REL_RR",
        "KURT_REL_RR", "SKEW_REL_RR", "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU",
        "HF", "HF_PCT", "HF_NU", "TP", "LF_HF", "HF_LF", "GSR_mean_gsr", "GSR_std_gsr",
        "GSR_max_mean_gsr_diff", "GSR_min_mean_gsr_diff", "GSR_gsr_mode", "GSR_gsr_skewness",
        "GSR_gsr_kurtosis", "GSR_gsr_max_mode_diff", "GSR_num_peaks", "GSR_time_between_peaks",
        "GSR_mean_first_derivative_gsr", "GSR_std_first_derivative_gsr", "GSR_mean_second_derivative_gsr",
        "GSR_std_second_derivative_gsr", "sampen", "higuci", "datasetId", "condition"]

cols2 = ["MEAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD",
         "pNN25", "pNN50", "SD1", "SD2", "KURT", "SKEW", "MEAN_REL_RR",
         "SDRR_REL_RR", "RMSSD_REL_RR", "SDSD_REL_RR", "SDRR_RMSSD_REL_RR",
         "KURT_REL_RR", "SKEW_REL_RR", "VLF", "VLF_PCT", "LF", "LF_PCT", "LF_NU",
         "HF", "HF_PCT", "HF_NU", "TP", "LF_HF", "HF_LF", "GSR_mean_gsr", "GSR_std_gsr",
         "GSR_max_mean_gsr_diff", "GSR_min_mean_gsr_diff", "GSR_gsr_mode", "GSR_gsr_skewness",
         "GSR_gsr_kurtosis", "GSR_gsr_max_mode_diff", "GSR_num_peaks", "GSR_time_between_peaks",
         "GSR_mean_first_derivative_gsr", "GSR_std_first_derivative_gsr", "GSR_mean_second_derivative_gsr",
         "GSR_std_second_derivative_gsr", "condition"]

dataframe = read_dataset("new_dataset.csv", cols)

# show_histograms(dataframe, cols2)

# create_jointplot(dataframe, cols2)

# create_pairplot(dataframe, cols2)

x_train, x_test, y_train, y_test = split_data(dataframe, cols2)

x_train, y_train = oversample_data(x_train, y_train)

x_test, y_test = oversample_data(x_test, y_test)

x_train, mean, std = scale_data(x_train)

x_test, _, _ = scale_data(x_test)

train_data = reshape_data(x_train, y_train)

test_data = reshape_data(x_test, y_test)

print(train_data.shape)
print(test_data.shape)

df_scaled = pd.DataFrame(x_train, columns=cols2[:-1])
df_scaled['condition'] = y_train
print(df_scaled.head())

model, predictions = k_nearest_neighbors(x_train, y_train, x_test)

# model, predictions = logistic_regression(x_train, y_train, x_test)

# model, predictions = gaussian_naive_bayes(x_train, y_train, x_test)

# model, predictions = support_vector_machines(x_train, y_train, x_test)

# model, predictions = decision_tree(x_train, y_train, x_test)

# model, predictions = random_forest(x_train, y_train, x_test)

if predictions is not None:
    evaluate_model(y_test, predictions)


ui = GUI(model, mean, std)
# ui.show_ui()
