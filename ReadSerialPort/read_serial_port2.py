import serial
import time
import numpy as np
import tkinter as tk
from data_utils import scale_data
from generate_features import calculate_features
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def read_extract_scale_predict(port, model, mean, std, ui):

    start_time = time.time()
    data_hr = []
    data_gsr = []

    # create the plots
    fig, (ax_hr, ax_gsr) = plt.subplots(ncols=2, figsize=(10, 5))
    ax_hr.set_xlabel('Time')
    ax_hr.set_ylabel('HR')
    ax_hr.set_title('HR Data')
    line_hr, = ax_hr.plot([], [], label='HR', color='red')
    ax_hr.legend()
    ax_gsr.set_xlabel('Time')
    ax_gsr.set_ylabel('GSR')
    ax_gsr.set_title('GSR Data')
    line_gsr, = ax_gsr.plot([], [], label='GSR', color='blue')
    ax_gsr.legend()

    # create the canvas to display the plots in the UI
    canvas = FigureCanvasTkAgg(fig, master=ui)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    output_label = tk.Label(ui, text="Analyzing...", font=("Helvetica", 16), width=1200)
    output_label.pack()

    ser = serial.Serial(port, 9600)
    while True:
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

                    # update the plots
                    x = np.arange(len(data_hr))
                    line_hr.set_data(x, data_hr)
                    ax_hr.relim()
                    ax_hr.autoscale_view()
                    line_gsr.set_data(x, data_gsr)
                    ax_gsr.relim()
                    ax_gsr.autoscale_view()
                    canvas.draw()

                except ValueError:
                    print("Invalid data received")
                    continue
        else:
            time.sleep(1)
        if len(data_hr) >= 60 and len(data_gsr) >= 60:
            print(len(data_hr))
            print(len(data_gsr))
            if len(data_hr) > 60 or len(data_gsr) > 60:
                data_hr = data_hr[-60:]
                data_gsr = data_gsr[-60:]
            features = calculate_features(data_hr, data_gsr)
            features = features.reshape(1, -1)
            print(features)
            features_scaled = scale_data(features, mean, std)
            for row in features_scaled:
                print([format(elem, ".3f") for elem in row])

            print("*****************************************************************")
            print(model.predict_proba(features_scaled))
            if model.predict(features_scaled) == 1:
                print("The subject is telling the truth")
                output_label.config(text="The subject is telling the truth", bg="green")
            else:
                print("The subject is lying")
                output_label.config(text="The subject is lying", bg="red")
    ser.close()
