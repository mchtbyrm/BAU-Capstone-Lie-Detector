import csv

import numpy as np
import serial
import time


def calculate_features(gender, bpm_data, gsr_data):
    # Calculate BPM features
    min_bpm = np.min(bpm_data)
    max_bpm = np.max(bpm_data)
    mean_bpm = np.mean(bpm_data)
    hrv = np.std(np.diff(bpm_data))
    hrv2 = (max_bpm - min_bpm) / mean_bpm
    print("-------------------------------------------------------------------------------")
    print("Hrv:", hrv)
    print("Hrv2:", hrv2)
    print("-------------------------------------------------------------------------------")

    # Calculate GSR features
    min_gsr = np.min(gsr_data)
    max_gsr = np.max(gsr_data)
    mean_gsr = np.mean(gsr_data)
    gsr_variability = np.std(gsr_data)

    # Return newly calculated features
    return np.array([[gender, min_bpm, max_bpm, mean_bpm, hrv, min_gsr, max_gsr, mean_gsr, gsr_variability]])


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
    read_serial()
