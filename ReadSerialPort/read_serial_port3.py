import csv

import numpy as np
import scipy
from scipy import stats, signal
import serial
import time
#Read csv file (there is 2 float numbers in each row seperated by comma) seperate and write them to 2 different lists

def read_csv(file):
    bpm_data = []
    gsr_data = []
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            bpm_data.append(float(row[0]))
            gsr_data.append(float(row[1]))
        return bpm_data, gsr_data

def extract_features(bpm_data, fs):
    # Calculate time-domain HRV metrics
    features = {}
    bpm_data = np.array(bpm_data)
    rr_intervals = 60000 / bpm_data
    # mean of RR intervals
    features['mean_rr'] = np.mean(rr_intervals)
    # standard deviation of RR intervals (SDNN)
    features['sdnn'] = np.std(rr_intervals)
    # root mean square of differences of successive RR intervals (RMSSD)
    features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    # percentage of successive RR interval differences greater than 50 ms (pNN50)
    features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100

    # Calculate frequency-domain HRV metrics
    f, Pxx = signal.welch(rr_intervals, fs=fs)
    lf = (0.04 <= f) & (f <= 0.15)
    hf = (0.15 < f) & (f <= 0.4)
    # power in LF band (0.04-0.15 Hz)
    features['lf'] = np.trapz(Pxx[lf], f[lf])
    # power in HF band (0.15-0.4 Hz)
    features['hf'] = np.trapz(Pxx[hf], f[hf])
    # LF/HF ratio
    features['lf_hf_ratio'] = features['lf'] / features['hf']

    return features


def calculate_features(gender):
    # Read BPM and GSR data
    bpm_data, gsr_data = read_csv("/Users/erguncan/BAU-Capstone-Lie-Detector/ReadSerialPort/yousif1684591776.1853743raw_data.csv")

    # Calculate HRV features
    hrv = extract_features(bpm_data, 1)
    # Calculate additional HRV features
    #hrv_mode = stats.mode(bpm_data)[0][0]  # Most frequent BPM value (Mode)




    # Calculate BPM features
    mean_bpm = np.mean(bpm_data)  # Mean of BPM
    std_bpm = np.std(bpm_data)  # Standard deviation of BPM
    max_mean_bpm_diff = np.abs(np.max(bpm_data) - mean_bpm)  # Difference between max BPM and mean BPM
    min_mean_bpm_diff = np.abs(np.min(bpm_data) - mean_bpm)  # Difference between min BPM and mean BPM

    # Calculate additional BPM features
    bpm_mode = stats.mode(bpm_data)[0][0]  # Most frequent BPM value (Mode)
    bpm_skewness = stats.skew(bpm_data)  # Skewness of BPM data
    bpm_kurtosis = stats.kurtosis(bpm_data)  # Kurtosis of BPM data
    bpm_max_mode_diff = np.abs(np.max(bpm_data) - bpm_mode)  # Difference between max BPM and mode

    # Calculate GSR features
    mean_gsr = np.mean(gsr_data)  # Mean of GSR
    std_gsr = np.std(gsr_data)  # Standard deviation of GSR
    max_mean_gsr_diff = np.abs(np.max(gsr_data) - mean_gsr)  # Difference between max GSR and mean GSR
    min_mean_gsr_diff = np.abs(np.min(gsr_data) - mean_gsr)  # Difference between min GSR and mean GSR

    # Calculate additional GSR features
    gsr_mode = stats.mode(gsr_data)[0][0]  # Most frequent GSR value (Mode)
    gsr_skewness = stats.skew(gsr_data)  # Skewness of GSR data
    gsr_kurtosis = stats.kurtosis(gsr_data)  # Kurtosis of GSR data
    gsr_max_mode_diff = np.abs(np.max(gsr_data) - gsr_mode)  # Difference between max GSR and mode

    # Calculate number of peaks and time between
    peaks, _ = scipy.signal.find_peaks(gsr_data)
    num_peaks = len(peaks)  # Number of peaks
    #avg_peak_height = np.mean(gsr_data[peaks]) if peaks.size > 0 else 0  # Average height of peaks
    time_between_peaks = np.mean(np.diff(peaks)) if peaks.size > 1 else 0  # Average time between peaks
    gsr_gradient = np.gradient(gsr_data)  # maybe this feature is not needed

    # Calculate cross-correlation of GSR and BPM data
    cross_corr = np.correlate(gsr_data - np.mean(gsr_data), bpm_data - np.mean(bpm_data), mode='valid')[0]

    # Derivative features

    first_derivative_bpm = np.diff(bpm_data)
    second_derivative_bpm = np.diff(bpm_data, n=2)  # n=2 for second derivative

    first_derivative_gsr = np.diff(gsr_data)
    second_derivative_gsr = np.diff(gsr_data, n=2)  # n=2 for second derivative

    mean_first_derivative_bpm = np.mean(first_derivative_bpm)
    std_first_derivative_bpm = np.std(first_derivative_bpm)

    mean_second_derivative_bpm = np.mean(second_derivative_bpm)
    std_second_derivative_bpm = np.std(second_derivative_bpm)

    mean_first_derivative_gsr = np.mean(first_derivative_gsr)
    std_first_derivative_gsr = np.std(first_derivative_gsr)

    mean_second_derivative_gsr = np.mean(second_derivative_gsr)
    std_second_derivative_gsr = np.std(second_derivative_gsr)

    # Print all features
    print("-------------------------------------------------------------------------------")
    print("HRV Features:")
    for key, value in hrv.items():
        print(key + ": ", value)
    print("-------------------------------------------------------------------------------")
    print("Gender: ", gender)
    print("Std BPM: ", std_bpm)
    print("BPM Skewness: ", bpm_skewness)
    print("BPM Kurtosis: ", bpm_kurtosis)
    print("BPM Max Mode Diff: ", bpm_max_mode_diff)
    print("-------------------------------------------------------------------------------")
    print("Std GSR: ", std_gsr)
    print("GSR Skewness: ", gsr_skewness)
    print("GSR Kurtosis: ", gsr_kurtosis)
    print("GSR Max Mode Diff: ", gsr_max_mode_diff)
    print("-------------------------------------------------------------------------------")
    print("Number of Peaks: ", num_peaks)
    #print("Avg Peak Height: ", avg_peak_height)
    print("Time Between Peaks: ", time_between_peaks)
    print("-------------------------------------------------------------------------------")
    print("Cross-Correlation: ", cross_corr)
    print("-------------------------------------------------------------------------------")
    print("Mean First Derivative BPM: ", mean_first_derivative_bpm)
    print("Std First Derivative BPM: ", std_first_derivative_bpm)
    print("-------------------------------------------------------------------------------")
    print("Mean Second Derivative BPM: ", mean_second_derivative_bpm)
    print("Std Second Derivative BPM: ", std_second_derivative_bpm)
    print("-------------------------------------------------------------------------------")
    print("Mean First Derivative GSR: ", mean_first_derivative_gsr)
    print("Std First Derivative GSR: ", std_first_derivative_gsr)
    print("-------------------------------------------------------------------------------")
    print("Mean Second Derivative GSR: ", mean_second_derivative_gsr)
    print("Std Second Derivative GSR: ", std_second_derivative_gsr)
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")



def read_serial(port='COM6', baudrate=9600):
    ser = serial.Serial(port, baudrate, timeout=1)
    name_of_subject = input("Enter the name of the subject: ")
    gender = input("Enter the gender of the subject (1 for Male and 0 for Female): ")
    bpm_data = []
    gsr_data = []
    start_time = time.time()
    end_time = start_time + 100  # if you want to run for 5 minutes, change this to 300
    while time.time() < end_time:
        if ser.in_waiting > 0:
            values = ser.readline().decode().strip().split(',')
            print(values)
            if len(values) == 4 and 0 < float(values[0]) < 1024 \
                    and 50 < float(values[1]) < 200 and start_time + 10 < time.time():
                try:
                    bpm = float(values[1])
                    gsr = float(values[0])
                    print(f"BPM: {bpm}, GSR: {gsr}")
                    bpm_data.append(bpm)
                    gsr_data.append(gsr)
                    csv_file = open(name_of_subject + str(start_time) + 'raw_data.csv', 'a', newline='')
                    writer = csv.writer(csv_file)
                    writer.writerow([bpm, gsr])
                    csv_file.close()
                except ValueError:
                    print("Invalid data")
        else:
            print("Waiting for data...")
            time.sleep(0.5)

    features = calculate_features(int(gender), bpm_data, gsr_data)
    gender_map = {1: 'Male', 0: 'Female'}
    gender = gender_map[features[0][0]]

    min_gsr = features[0][1]
    max_gsr = features[0][2]
    mean_gsr = features[0][3]
    gsr_variability = features[0][4]

    min_bpm = features[0][5]
    max_bpm = features[0][6]
    mean_bpm = features[0][7]
    hrv = features[0][8]

    csv_file = open('features.csv', 'a', newline='')  # 'a' means append. so it will not overwrite the file
    writer = csv.writer(csv_file)
    writer.writerow([gender, min_bpm, max_bpm, mean_bpm, hrv, min_gsr, max_gsr, mean_gsr, gsr_variability])
    csv_file.close()


if __name__ == '__main__':
    #read_serial()
    calculate_features(1)
