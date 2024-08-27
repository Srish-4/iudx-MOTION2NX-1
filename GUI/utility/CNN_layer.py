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
from utility import cifar10
from utility import result_Cifar10
from utility import Six_layer
import subprocess
import os

selected_image_path = ""

base_path = os.getenv("BASE_DIR", default=None)


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
        # 2 for selecting CNN and cifar 10 dataset
        loading.get_user_choice(2)
        loading.choose_dataset(2)
        selected_image_path = self.image_path
        if self.image_path == base_path+"/GUI/aeroplane.png":
            window.destroy()
            cifar10.call(1)
            # 1 for aeroplane
        elif self.image_path == base_path+"/GUI/automobile.png":
            window.destroy()
            cifar10.call(2)
            # 2 for automobile
        elif self.image_path == base_path+"/GUI/bird.png":
            window.destroy()
            cifar10.call(3)
            # 3 for bird
        elif self.image_path == base_path+"/GUI/cat.png":
            window.destroy()
            cifar10.call(4)
            # 4 for cat
        elif self.image_path == base_path+"/GUI/deer.png":
            window.destroy()
            cifar10.call(5)
            # 5 for deer
        elif self.image_path == base_path+"/GUI/dog.png":
            window.destroy()
            cifar10.call(6)
            # 6 for dog
        elif self.image_path == base_path+"/GUI/frog.png":
            window.destroy()
            cifar10.call(7)
            # 7 for frog
        elif self.image_path == base_path+"/GUI/horse.png":
            window.destroy()
            cifar10.call(8)
            # 8 for horse
        elif self.image_path == base_path+"/GUI/ship.png":
            window.destroy()
            cifar10.call(9)
            # 9 for ship
        elif self.image_path == base_path+"/GUI/truck.png":
            window.destroy()
            cifar10.call(10)
            # 10 for truck

        # Open a dialog to select the CSV file


def call():
    global window
    window = tk.Tk()
    window.title("SMPC")

    WIDTH = 800
    HEIGHT = 600

    window.geometry("1500x720")

    canvas_2 = Canvas(window, width=900, height=0)
    canvas_2.pack(side=TOP, fill=X)
    canvas_2.option_add("*TCombobox*Listbox.font", "sans 20 bold")
    canvas_2.option_add("*TCombobox*Listbox.justify", "center")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure('W.TCombobox', arrowsize=25)

    options = [
        "Secure Multiparty Computation",
        "Neural Network Inferencing",
        "Neural Network Inferencing with helper node",
        "Setup: Two Layer Neural Network",
        "Setup: Five Layer Neural Network",
        # "Convolution Neural Network",
        "Convolution Neural Network Inferencing with helper node",
        "Convolution Neural Network Split",
        "Setup: Four Layer Convolution Network",
        "Setup: Six Layer Convolution Network"
    ]

    def display_selected(event):
        choice = my_combo.get()
        if choice == "Secure Multiparty Computation":
            window.destroy()
            SMPC.call()
        elif choice == "Neural Network Inferencing":
            window.destroy()
            NN.call()
        elif choice == "Neural Network Inferencing with helper node":
            window.destroy()
            NN_helper.call()
        elif choice == "Setup: Two Layer Neural Network":
            window.destroy()
            two_layer.call()
        # elif choice == "Convolution Neural Network":
         #   window.destroy()
            #  CNN_layer.call()
        elif choice == "Convolution Neural Network Inferencing with helper node":
            window.destroy()
            CNN_helper.call()
        elif choice == "Convolution Neural Network Split":
            window.destroy()
            CNN_Split.call()
        elif choice == "Setup: Five Layer Neural Network":
            window.destroy()
            five_layer.call()
        elif choice == "Setup: Four Layer Convolution Network":
            window.destroy()
            four_layer.call()
        elif choice == "Setup: Six Layer Convolution Network":
            window.destroy()
            Six_layer.call()

    clicked = StringVar()
    my_combo = ttk.Combobox(canvas_2, values=options, font=(
        'sans 20 bold'), justify=CENTER, textvariable=clicked, style='W.TCombobox', state="readonly", width=30)
    my_combo.grid(row=0, column=0, columnspan=4, ipadx=200,
                  ipady=10, padx=(220, 10), pady=(15, 10), sticky="ew")
    my_combo.set("Convolution Neural Network - CIFAR 10")
    my_combo.bind("<<ComboboxSelected>>", display_selected)

    image_paths_uploaded = [
        base_path+"/GUI/aeroplane.png",
        base_path+"/GUI/automobile.png",
        base_path+"/GUI/bird.png",
        base_path+"/GUI/cat.png",
        base_path+"/GUI/deer.png",
    ]

    num_columns_uploaded = len(image_paths_uploaded)
    available_width = WIDTH - 20
    available_height = 280

    max_width_uploaded = (available_width // num_columns_uploaded)
    max_height_uploaded = available_height
    max_size_uploaded = max(
        available_width // num_columns_uploaded, available_height)

    uploaded_frame = Frame(window)
    uploaded_frame.pack(side=TOP, fill=X, pady=2)

    image_paths_additional = [
        base_path+"/GUI/dog.png",
        base_path+"/GUI/frog.png",
        base_path+"/GUI/horse.png",
        base_path+"/GUI/ship.png",
        base_path+"/GUI/truck.png",
    ]

    available_width = WIDTH - 20
    available_height = 200
    num_columns_additional = len(image_paths_additional)

    uploaded_frame = Frame(window)
    uploaded_frame.pack(side=TOP, fill=X, pady=10)

    for i, image_path in enumerate(image_paths_uploaded):
        image = Image.open(image_path)
        image = image.resize(
            (max_size_uploaded, max_size_uploaded), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(uploaded_frame, image=photo_image)
        label.image = photo_image
        label.grid(row=0, column=i, padx=5, pady=5)
        image_radio_button = ImageRadioButton(
            uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event,
                   rb=image_radio_button: select_radio_button(rb))

    additional_frame = Frame(window)
    additional_frame.pack(side=TOP, fill=X, pady=10)

    for i, image_path in enumerate(image_paths_additional):
        image = Image.open(image_path)
        image = image.resize(
            (max_size_uploaded, max_size_uploaded), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(additional_frame, image=photo_image)
        label.image = photo_image
        label.grid(row=0, column=i, padx=5, pady=5)
        image_radio_button = ImageRadioButton(
            uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event,
                   rb=image_radio_button: select_radio_button(rb))

    window.resizable(False, False)
    window.mainloop()


def select_radio_button(rb):
    rb.variable.set(rb.value)
    rb.on_select()
