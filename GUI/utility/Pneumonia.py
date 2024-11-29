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
        loading.get_user_choice(3)
        loading.choose_dataset(3)
        window.destroy()
        File_location.call(selected_image_path)
        # Open a dialog to select the CSV file


def call():
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

    image_paths_uploaded = [
        base_path+"/GUI/pneumonia_test_data/P1.jpeg",
        base_path+"/GUI/pneumonia_test_data/P1.jpeg",
        base_path+"/GUI/pneumonia_test_data/P1.jpeg"
    ]
    image_paths_additional = [
        base_path+"/GUI/pneumonia_test_data/P1.jpeg",
        base_path+"/GUI/pneumonia_test_data/P1.jpeg",
        base_path+"/GUI/pneumonia_test_data/P1.jpeg"
    ]

    image_paths_additional1 = [
        base_path+"/GUI/pneumonia_test_data/P1.jpeg",
        base_path+"/GUI/pneumonia_test_data/P1.jpeg",
        base_path+"/GUI/pneumonia_test_data/P1.jpeg"
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
