import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# create a tinker window
root = tk.Tk()

# set size of the window
root.geometry("600x400")

# create a label
time_label = tk.Label(root, text="The person has not started the test yet.")
status_label = tk.Label(root, text="Machine is not working.")

# set the position of the label
time_label.pack(pady=20)
status_label.pack()

# define a function to be called when the button is clicked
seconds = [0]
minutes = [0]
is_running = [False]


def on_button_click():
    if not is_running[0]:
        is_running[0] = True
        time_label.config(text="The person has been in the polygraph test for {minutes[0]:02d}:{seconds[0]:02d}")
        status_label.config(text="Machine started to work")

        def update_time():
            if is_running[0]:
                seconds[0] += 1
                if seconds[0] == 60:
                    seconds[0] = 0
                    minutes[0] += 1
                time_label.config(text=f"The person has been in the polygraph test for {minutes[0]:02d}:{seconds[0]:02d}")
                if minutes[0] == 0 and seconds[0] == 10:
                    result = random.choice(["True", "False"])
                    plt.clf()
                    plt.bar(["True", "False"], [result.count("True"), result.count("False")])
                    canvas.draw()
                    if result == "True":
                        status_label.config(text="The person is telling the truth.")
                    else:
                        status_label.config(text="The person is lying.")
                root.after(1000, update_time)
        update_time()
    else:
        is_running[0] = False
        time_label.config(text=f"The person was tested for {minutes[0]:02d}:{seconds[0]:02d}")
        status_label.config(text="Machine stopped working")

# create a button
button = tk.Button(root, text="Start Test", command=on_button_click)

# set the position of the button
button.pack(pady=10)

# create a matplotlib figure
fig = plt.Figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
ax.set_ylim([0, 1])
ax.set_ylabel("Probability")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

# start the main loop
root.mainloop()
