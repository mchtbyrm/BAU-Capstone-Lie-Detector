import sadkasd
import time
import random


def send_data():
    ser = serial.Serial('COM2', 9600)

    while True:
        heart_rate = random.randint(60, 190)  # Generate a random heart rate
        gsr = random.uniform(1, 10)  # Generate a random GSR value

        data = f"{heart_rate},{gsr}\n".encode()  # Data is encoded to bytes
        ser.write(data)  # Write the data to the serial port

        time.sleep(5)  # Sleep 5 seconds


if __name__ == '__main__':
    send_data()
