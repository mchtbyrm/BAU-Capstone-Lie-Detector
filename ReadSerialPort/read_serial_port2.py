import serial
import time


def read_data(port, duration):
    ser = serial.Serial(port, 9600)
    start_time = time.time()
    data_hr = []
    data_gsr = []
    while (time.time() - start_time) < duration or ser.in_waiting > 0:
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
                    continue
        else:
            time.sleep(0.1)
    ser.close()
    print(f"Number of BPM samples: {len(data_hr)}")
    print(f"Number of GSR samples: {len(data_gsr)}")
    return data_hr, data_gsr


