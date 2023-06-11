import serial.tools.list_ports


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(
            f"Device: {p.device}, Name: {p.name}, Description: {p.description}, HWID: {p.hwid}, VID: {p.vid}, PID: {p.pid}, Serial Number: {p.serial_number}, Manufacturer: {p.manufacturer}")



if __name__ == '__main__':
    list_ports()
