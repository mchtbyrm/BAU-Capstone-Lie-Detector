import threading
import tkinter as tk
from ReadSerialPort.read_serial_port2 import read_extract_scale_predict


class GUI:
    def __init__(self, model, mean, std):
        self.root = tk.Tk()
        self.gender = 0
        self.model = model
        self.mean = mean
        self.std = std

    def start_clicked(self):
        print("Start button clicked")
        # run the read_extract_scale_predict function in a separate thread
        t = threading.Thread(target=read_extract_scale_predict,
                             args=(self.gender, 'COM1', self.model, self.mean, self.std, self.root))
        t.start()

    def set_gender(self, value):
        self.gender = value

    def show_ui(self):
        self.root.geometry("1000x650")

        # choose gender of subject
        gender_label = tk.Label(self.root, text="Select gender of subject:")
        gender_label.pack()

        male_button = tk.Radiobutton(self.root, text="Male", value=1, command=lambda: self.set_gender(1))
        male_button.pack()

        female_button = tk.Radiobutton(self.root, text="Female", value=0, command=lambda: self.set_gender(0))
        female_button.pack()

        # click start button to call start_clicked function
        start_button = tk.Button(self.root, text="Start", command=self.start_clicked)
        start_button.pack()

        self.root.mainloop()
