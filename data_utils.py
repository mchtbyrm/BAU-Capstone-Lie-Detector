import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# read the dataset from the csv file
def read_dataset(file, cols):
    df = pd.read_csv(file, names=cols)
    # sklearn does not work with strings. it only works with numbers.
    df["Gender"] = (df["Gender"] == "Male").astype(int)  # convert to 0 or 1. If Gender is Male then 1, else 0
    df["Class"] = (df["Class"] == "Truth").astype(int)  # convert to 0 or 1. If Class is Truth then 1, else 0
    print(df.head())  # print the first 5 rows
    print(df.tail())  # print the last 5 rows
    print(df.info())  # print the information about the dataframe
    print(df.describe())  # print the statistical information about the dataframe
    return df


# show the histograms of the dataset for each feature
def show_histograms(df, cols):
    for label in cols[:-1]:
        plt.hist(df[df["Class"] == 1][label], color='blue', label='Truth', alpha=0.7, density=True)
        plt.hist(df[df["Class"] == 0][label], color='red', label='Lie', alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()
        plt.show()


# split the dataset into training and testing sets
def split_data(df, cols):
    x = df[cols[:-1]].values
    y = df[cols[-1]].values
    # test_size=0.3 means 30% of the data will be used for testing and 70% for training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


# oversample the data. this is important because the dataset is imbalanced
# for example, there are 1000 rows of data and 900 of them are Truth and 100 of them are Lie
# oversampling will duplicate the rows of the Lie data so that the number of rows of Truth and Lie are equal
def oversample_data(x, y):
    # random_state=0 means the random numbers will be the same every time
    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_resample(x, y)
    return x_resampled, y_resampled


# scale the data. this is important because the features have different ranges
# for example, the range of the values of the one feature is 0-100 and
# the range of the values of the other feature is 0-1000
def scale_data(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled
