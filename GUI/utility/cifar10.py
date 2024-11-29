from tkinter import *
from tkinter import ttk
from utility import paint
import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfile
from utility import SMPC
from utility import NN
from utility import loading
from utility import File_location
from utility import CNN_helper
from utility import CNN_layer
from utility import NN_helper
from utility import two_layer
from utility import five_layer
from utility import CNN_Split
from utility import four_layer
import subprocess
import os


selected_image_path = ""

base_path = os.getenv("BASE_DIR")


class ImageRadioButton(Frame):
    def __init__(self, master=None, image_path="", value=None, class_label="", csv_files=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.value = value
        self.variable = IntVar()
        self.image_path = image_path
        self.class_label = class_label

    def create_widgets(self):
        self.variable.set(0)
        image = Image.open(self.image_path)
        image = image.resize((50, 50), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(image)
        self.radio_button = Radiobutton(
            self, image=self.image, variable=self.variable, value=self.value, indicatoron=0, command=self.on_select)
        self.radio_button.image = self.image
        self.radio_button.pack(padx=5, pady=5)

    def on_select(self):
        global selected_image_path
        selected_image_path = self.image_path
        window.destroy()
        File_location.call(selected_image_path)
        # Open a dialog to select the CSV file


def call(value):
    global window
    window = tk.Tk()
    window.title("SMPC")

    WIDTH = 800
    HEIGHT = 600

    window.geometry("1300x850")

    canvas_2 = Canvas(window, width=900, height=0)
    canvas_2.pack(side=TOP, fill=X)
    canvas_2.option_add("*TCombobox*Listbox.font", "sans 20 bold")
    uploaded_frame = Frame(window)
    uploaded_frame.pack(side=TOP, fill=X, pady=2)

    if value == 1:

        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/0_1.png",
            base_path+"/GUI/cifar_test_data/0_2.png",
            base_path+"/GUI/cifar_test_data/0_3.png",
            base_path+"/GUI/cifar_test_data/0_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/0_4.png",
            base_path+"/GUI/cifar_test_data/0_5.png",
            base_path+"/GUI/cifar_test_data/0_6.png",
            base_path+"/GUI/cifar_test_data/0_11.png",
        ]

        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/0_7.png",
            base_path+"/GUI/cifar_test_data/0_8.png",
            base_path+"/GUI/cifar_test_data/0_9.png",
            base_path+"/GUI/cifar_test_data/0_12.png",
        ]
    elif value == 2:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/1_1.png",
            base_path+"/GUI/cifar_test_data/1_2.png",
            base_path+"/GUI/cifar_test_data/1_3.png",
            base_path+"/GUI/cifar_test_data/1_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/1_4.png",
            base_path+"/GUI/cifar_test_data/1_5.png",
            base_path+"/GUI/cifar_test_data/1_6.png",
            base_path+"/GUI/cifar_test_data/1_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/1_7.png",
            base_path+"/GUI/cifar_test_data/1_8.png",
            base_path+"/GUI/cifar_test_data/1_9.png",
            base_path+"/GUI/cifar_test_data/1_12.png",
        ]

    elif value == 3:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/2_1.png",
            base_path+"/GUI/cifar_test_data/2_2.png",
            base_path+"/GUI/cifar_test_data/2_3.png",
            base_path+"/GUI/cifar_test_data/2_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/2_4.png",
            base_path+"/GUI/cifar_test_data/2_5.png",
            base_path+"/GUI/cifar_test_data/2_6.png",
            base_path+"/GUI/cifar_test_data/2_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/2_7.png",
            base_path+"/GUI/cifar_test_data/2_8.png",
            base_path+"/GUI/cifar_test_data/2_9.png",
            base_path+"/GUI/cifar_test_data/2_12.png",
        ]
    elif value == 4:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/3_1.png",
            base_path+"/GUI/cifar_test_data/3_2.png",
            base_path+"/GUI/cifar_test_data/3_3.png",
            base_path+"/GUI/cifar_test_data/3_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/3_4.png",
            base_path+"/GUI/cifar_test_data/3_5.png",
            base_path+"/GUI/cifar_test_data/3_6.png",
            base_path+"/GUI/cifar_test_data/3_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/3_7.png",
            base_path+"/GUI/cifar_test_data/3_8.png",
            base_path+"/GUI/cifar_test_data/3_9.png",
            base_path+"/GUI/cifar_test_data/3_12.png",
        ]
    elif value == 5:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/4_1.png",
            base_path+"/GUI/cifar_test_data/4_2.png",
            base_path+"/GUI/cifar_test_data/4_3.png",
            base_path+"/GUI/cifar_test_data/4_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/4_4.png",
            base_path+"/GUI/cifar_test_data/4_5.png",
            base_path+"/GUI/cifar_test_data/4_6.png",
            base_path+"/GUI/cifar_test_data/4_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/4_7.png",
            base_path+"/GUI/cifar_test_data/4_8.png",
            base_path+"/GUI/cifar_test_data/4_9.png",
            base_path+"/GUI/cifar_test_data/4_12.png",
        ]
    elif value == 6:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/5_1.png",
            base_path+"/GUI/cifar_test_data/5_2.png",
            base_path+"/GUI/cifar_test_data/5_3.png",
            base_path+"/GUI/cifar_test_data/5_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/5_4.png",
            base_path+"/GUI/cifar_test_data/5_5.png",
            base_path+"/GUI/cifar_test_data/5_6.png",
            base_path+"/GUI/cifar_test_data/5_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/5_7.png",
            base_path+"/GUI/cifar_test_data/5_8.png",
            base_path+"/GUI/cifar_test_data/5_9.png",
            base_path+"/GUI/cifar_test_data/5_12.png",
        ]
    elif value == 7:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/6_1.png",
            base_path+"/GUI/cifar_test_data/6_2.png",
            base_path+"/GUI/cifar_test_data/6_3.png",
            base_path+"/GUI/cifar_test_data/6_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/6_4.png",
            base_path+"/GUI/cifar_test_data/6_5.png",
            base_path+"/GUI/cifar_test_data/6_6.png",
            base_path+"/GUI/cifar_test_data/6_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/6_7.png",
            base_path+"/GUI/cifar_test_data/6_8.png",
            base_path+"/GUI/cifar_test_data/6_9.png",
            base_path+"/GUI/cifar_test_data/6_12.png",
        ]
    elif value == 8:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/7_1.png",
            base_path+"/GUI/cifar_test_data/7_2.png",
            base_path+"/GUI/cifar_test_data/7_3.png",
            base_path+"/GUI/cifar_test_data/7_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/7_4.png",
            base_path+"/GUI/cifar_test_data/7_5.png",
            base_path+"/GUI/cifar_test_data/7_6.png",
            base_path+"/GUI/cifar_test_data/7_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/7_7.png",
            base_path+"/GUI/cifar_test_data/7_8.png",
            base_path+"/GUI/cifar_test_data/7_9.png",
            base_path+"/GUI/cifar_test_data/7_12.png",
        ]
    elif value == 9:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/8_1.png",
            base_path+"/GUI/cifar_test_data/8_2.png",
            base_path+"/GUI/cifar_test_data/8_3.png",
            base_path+"/GUI/cifar_test_data/8_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/8_4.png",
            base_path+"/GUI/cifar_test_data/8_5.png",
            base_path+"/GUI/cifar_test_data/8_6.png",
            base_path+"/GUI/cifar_test_data/8_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/8_7.png",
            base_path+"/GUI/cifar_test_data/8_8.png",
            base_path+"/GUI/cifar_test_data/8_9.png",
            base_path+"/GUI/cifar_test_data/8_12.png",
        ]
    elif value == 10:
        image_paths_uploaded = [
            base_path+"/GUI/cifar_test_data/9_1.png",
            base_path+"/GUI/cifar_test_data/9_2.png",
            base_path+"/GUI/cifar_test_data/9_3.png",
            base_path+"/GUI/cifar_test_data/9_10.png",
        ]
        image_paths_additional = [
            base_path+"/GUI/cifar_test_data/9_4.png",
            base_path+"/GUI/cifar_test_data/9_5.png",
            base_path+"/GUI/cifar_test_data/9_6.png",
            base_path+"/GUI/cifar_test_data/9_11.png",
        ]
        image_paths_additional1 = [
            base_path+"/GUI/cifar_test_data/9_7.png",
            base_path+"/GUI/cifar_test_data/9_8.png",
            base_path+"/GUI/cifar_test_data/9_9.png",
            base_path+"/GUI/cifar_test_data/9_12.png",
        ]

    num_columns_uploaded = len(image_paths_uploaded)
    available_width = 1200 - 20
    available_height = 140

    max_width_uploaded = (available_width // num_columns_uploaded)
    max_height_uploaded = available_height
    max_size_uploaded = max(
        available_width // num_columns_uploaded, available_height)

    for i, image_path in enumerate(image_paths_uploaded):
        image = Image.open(image_path)
        image = image.resize((max_size_uploaded, 240), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(uploaded_frame, image=photo_image)
        label.image = photo_image
        label.grid(row=0, column=i, padx=(10, 20), pady=5)
        image_radio_button = ImageRadioButton(
            uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event,
                   rb=image_radio_button: select_radio_button(rb))

    additional_frame = Frame(window)
    additional_frame.pack(side=TOP, fill=X, pady=10)

    for i, image_path in enumerate(image_paths_additional):
        image = Image.open(image_path)
        image = image.resize((max_size_uploaded, 240), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(additional_frame, image=photo_image)
        label.image = photo_image
        label.grid(row=0, column=i, padx=(10, 20), pady=5)
        image_radio_button = ImageRadioButton(
            uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event,
                   rb=image_radio_button: select_radio_button(rb))

    additional1_frame = Frame(window)
    additional1_frame.pack(side=TOP, fill=X, pady=10)
    for i, image_path in enumerate(image_paths_additional1):
        image = Image.open(image_path)
        image = image.resize((max_size_uploaded, 240), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(additional1_frame, image=photo_image)
        label.image = photo_image
        label.grid(row=0, column=i, padx=(10, 20), pady=5)
        image_radio_button = ImageRadioButton(
            uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event,
                   rb=image_radio_button: select_radio_button(rb))

    window.resizable(False, False)
    window.mainloop()


def select_radio_button(rb):
    rb.variable.set(rb.value)
    rb.on_select()
