from ReadSerialPort.read_serial_port import read_serial
from data_utils import *


cols = ["Gender", "Min_BPM", "Max_BPM", "Mean_BPM", "HRV", "Min_GSR", "Max_GSR", "Mean_GSR", "GSR_Variability", "Class"]

dataframe = read_dataset("lie.csv", cols)

show_histograms(dataframe, cols)

# read_serial()


