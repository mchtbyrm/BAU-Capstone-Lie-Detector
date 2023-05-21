import serial
import time
import numpy as np
import random


def send_data():
    ser = serial.Serial('COM2', 9600)

    while True:
        if np.random.choice([True, False], p=[0.1, 0.9]):
            data = "nan nan nan nan\n"
        else:
            bpm = random.randint(60, 170)  # Generate a random heart rate
            gsr = random.uniform(0, 2)  # Generate a random GSR value
            systolic_bp = random.randint(300, 800)
            diastolic_bp = random.randint(300, 800)
            data = f"{gsr},{bpm},{systolic_bp},{diastolic_bp}\n"
        ser.write(data.encode())  # Write the data to the serial port

        time.sleep(1)  # Sleep for 1 second


if __name__ == '__main__':
    send_data()
