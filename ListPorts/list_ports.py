import serial.tools.list_ports


def list_all_available_serial_ports():
    # List available serial ports
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(port.device)


if __name__ == '__main__':
    list_all_available_serial_ports()
