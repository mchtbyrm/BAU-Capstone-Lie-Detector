import csv
import serial


# Change the port and baudrate to match your Arduino's serial settings
# port is the serial port your Arduino is connected to
# baudrate refers to the speed at which data is transferred and is usually 9600
def read_serial(port='COM1', baudrate=9600):
    ser = serial.serial_for_url(port, baudrate, timeout=1)
    while True:
        line = ser.readline().decode('utf-8').strip()
        print(line)
        if line:
            name, value1, value2 = line.split(',')
            csv_file = open(name + '.csv', 'a', newline='')  # 'a' means append. so it will not overwrite the file
            writer = csv.writer(csv_file)
            writer.writerow([value1, value2])
            csv_file.close()
