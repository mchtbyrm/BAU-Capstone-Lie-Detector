import random
import sadkasd
import time


# This method will send random values to the serial port
def send_random_values_to_serial():
    ser = serial.Serial('COM2', 9600, timeout=1)
    names = ['1', '2', '3', '4', '5']
    while True:
        name = random.choice(names)
        value1 = random.uniform(0, 10)
        value2 = random.uniform(0, 10)
        line = f"{name},{value1},{value2}\n"
        ser.write(line.encode()) 
        print(line)
        time.sleep(1)


if __name__ == '__main__':
    send_random_values_to_serial()
