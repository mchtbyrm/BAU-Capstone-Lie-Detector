import serial
import time
from data_utils import scale_data
from generate_features import calculate_features


def read_extract_scale_predict(port, model, mean, std):
    ser = serial.Serial(port, 9600)
    data_hr = []
    data_gsr = []
    while True:
        if ser.in_waiting > 0:
            read_serial = ser.readline()
            values = read_serial.decode().strip().split(',')
            print(values)
            if len(values) == 2:
                try:
                    hr = float(values[0])
                    gsr = float(values[1])
                    print(f"HR: {hr}, GSR: {gsr}")
                    data_hr.append(hr)
                    data_gsr.append(gsr)
                except ValueError:
                    print("Invalid data received")
                    continue
        else:
            time.sleep(5)
        if len(data_hr) >= 5 and len(data_gsr) >= 5:
            print(data_hr)
            print(data_gsr)
            features = calculate_features(data_hr, data_gsr)
            features = features.reshape(1, -1)
            print(features)
            features_scaled = scale_data(features, mean, std)
            for row in features_scaled:
                print([format(elem, ".3f") for elem in row])

            print("*****************************************************************")

            if model.predict(features_scaled) == 1:
                print("The subject is telling the truth")
            else:
                print("The subject is lying")
    ser.close()
