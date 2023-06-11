import threading
import time
import tkinter as tk

import numpy as np
import serial
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_utils import scale_data
from generate_features import calculate_features


class GUI:
    def __init__(self, model, mean, std):
        self.root = tk.Tk()
        self.model = model
        self.mean = mean
        self.std = std
        self.running = False  # Flag to indicate if the test is running or not

        self.root.geometry("1000x650")

        info_label = tk.Label(self.root, text="Please connect the Arduino to the computer")
        info_label.pack()

        info_label2 = tk.Label(self.root, text="To start the test, click the start button")
        info_label2.pack()

        self.start_button = tk.Button(self.root, text="Start", command=self.start_stop_clicked)
        self.start_button.pack()

        self.fig = plt.figure(figsize=(10, 5))
        self.ax_hr = self.fig.add_subplot(1, 2, 1)
        self.ax_gsr = self.fig.add_subplot(1, 2, 2)
        self.ax_hr.set_xlabel('Time')
        self.ax_hr.set_ylabel('HR')
        self.ax_hr.set_title('HR Data')
        self.line_hr, = self.ax_hr.plot([], [], label='HR', color='red')
        self.ax_hr.legend()
        self.ax_gsr.set_xlabel('Time')
        self.ax_gsr.set_ylabel('GSR')
        self.ax_gsr.set_title('GSR Data')
        self.line_gsr, = self.ax_gsr.plot([], [], label='GSR', color='blue')
        self.ax_gsr.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.output_label = tk.Label(self.root, text="Analyzing...", font=("Helvetica", 16), width=1200)
        self.output_label.pack()

        self.root.mainloop()

    def find_arduino_port(self):
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino" in p.manufacturer:  # check for the Arduino name in the manufacturer field
                return p.device
        return 'COM6'

    def start_stop_clicked(self):
        if not self.running:
            print("Start button clicked")
            port = self.find_arduino_port()
            print(f"Arduino port: {port}")
            self.running = True  # set the flag to True to indicate that the test is running
            self.start_button.config(text="Stop")
            self.reset_plots()  # reset the plots
            self.output_label.config(text="Analyzing...", bg="gray")  # reset the output label
            t = threading.Thread(target=self.read_extract_scale_predict, args=(port,))
            t.start()
        else:
            print("Stop button clicked")
            self.running = False  # set the flag to False to indicate that the test is not running
            self.start_button.config(text="Start")

    def reset_plots(self):
        self.line_hr.set_data([], [])
        self.line_gsr.set_data([], [])
        self.ax_hr.relim()
        self.ax_hr.autoscale_view()
        self.ax_gsr.relim()
        self.ax_gsr.autoscale_view()
        self.canvas.draw()

    def update_plots(self, data_hr, data_gsr):
        x = np.arange(len(data_hr))
        self.line_hr.set_data(x, data_hr)
        self.ax_hr.relim()
        self.ax_hr.autoscale_view()
        self.line_gsr.set_data(x, data_gsr)
        self.ax_gsr.relim()
        self.ax_gsr.autoscale_view()
        self.canvas.draw()

    def read_extract_scale_predict(self, port):
        start_time = time.time()
        data_hr = []
        data_gsr = []
        update_interval = 20  # Number of rows after which you want to update the model
        row_counter = 0  # Initialize row counter

        ser = serial.Serial(port, 9600)
        while self.running:  # loop while the test is running
            if ser.in_waiting > 0:
                read_serial = ser.readline()
                values = read_serial.decode().strip().split(',')
                print(values)
                if len(values) == 4 and start_time + 10 < time.time() and float(values[0]) > 0 and float(values[1]) > 0:
                    try:
                        gsr = float(values[0])
                        hr = float(values[1])
                        print(f"HR: {hr}, GSR: {gsr}")
                        data_hr.append(hr)
                        data_gsr.append(gsr)

                        self.update_plots(data_hr, data_gsr)

                    except ValueError:
                        print("Invalid data received")
                        continue
            else:
                time.sleep(1)

            row_counter += 1  # Increment row counter

            # Update the model if the row counter has reached the update interval
            if row_counter >= update_interval:
                row_counter = 0  # Reset row counter
                print(len(data_hr))
                print(len(data_gsr))
                if len(data_hr) > 40 or len(data_gsr) > 40:
                    data_hr = data_hr[-40:]
                    data_gsr = data_gsr[-40:]
                features = calculate_features(data_hr, data_gsr)
                features = features.reshape(1, -1)
                print(features)
                features_scaled = scale_data(features, self.mean, self.std)
                for row in features_scaled:
                    print([format(elem, ".3f") for elem in row])

                print("*****************************************************************")
                print(self.model.predict_proba(features_scaled))
                if self.model.predict(features_scaled) == 1:
                    print("The subject is telling the truth")
                    self.output_label.config(text="The subject is telling the truth", bg="green")
                else:
                    print("The subject is lying")
                    self.output_label.config(text="The subject is lying", bg="red")
        ser.close()

