import serial
import time
import random


def send_data():
    ser = serial.Serial('COM2', 9600)
    start_time = time.time()
    end_time = start_time + 60  # 60 seconds from now

    while time.time() < end_time:
        heart_rate = random.randint(60, 120)  # Generate a random heart rate
        gsr = random.uniform(0, 1)  # Generate a random GSR value

        data = f"{heart_rate},{gsr}\n".encode()  # Data is encoded to bytes
        ser.write(data)  # Write the data to the serial port

        time.sleep(5)  # Sleep 5 seconds

    ser.close()  # Close the serial port


if __name__ == '__main__':
    send_data()
