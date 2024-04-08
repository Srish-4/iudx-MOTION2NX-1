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
import subprocess
import os

selected_image_path=""

class ImageRadioButton(Frame):
    def __init__(self, master=None, image_path="", value=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.value = value
        self.variable = IntVar()
        self.image_path = image_path
        self.create_widgets()

    def create_widgets(self):
        #print("Creating widget for image:", self.image_path)  # Debugging print statement
        self.variable.set(0)
        image = Image.open(self.image_path)
        image = image.resize((50,50), Image.ANTIALIAS)
        #print("Image size:", image.size) 
        self.image = ImageTk.PhotoImage(image)
        self.radio_button = Radiobutton(self, image=self.image, variable=self.variable, value=self.value, indicatoron=0, command=self.on_select)
        self.radio_button.image = self.image  # To prevent garbage collection
        self.radio_button.pack(padx=5, pady=5) 
       

    def on_select(self):
        global selected_image_path
        selected_image_path = self.image_path        
       # print("Selected image:", self.image_path)
        update_selected_image()
        window.destroy()
        File_location.call(selected_image_path)
        # Redirect to another page or perform other actions based on the selected image
    def update_selected_image():
      global selected_image_path
      #print("Updating selected image:", selected_image_path)
      # You can perform any action here when an image is selected   

# window = Tk()
    def on_select(self):
        global selected_image_path
        selected_image_path = self.image_path        
       # print("Selected image:", self.image_path)
        window.destroy()
        File_location.call(selected_image_path)
        
    def select_radio_button(self, event):
        # When an image is clicked, set the associated radio button as selected
        self.variable.set(self.value)

     


def call():
    global window
    window = tk.Tk()
    window.title("SMPC")
    

    WIDTH = 800
    HEIGHT = 600

    window.geometry("900x560")

    canvas_2 = Canvas(window, width= WIDTH, height=70)
    canvas_2.pack(side=TOP, fill=X)
    canvas_2.option_add("*TCombobox*Listbox.font", "sans 20 bold")
    canvas_2.option_add("*TCombobox*Listbox.justify", "center")


    style = ttk.Style()
    style.theme_use("clam")
    style.configure('W.TCombobox', arrowsize = 25)

    options = [
        "Secure Multiparty Computation",
        "Neural Network Inferencing",
        "Neural Network Inferencing with helper node",
        "Setup: Two Layer Neural Network",
        "Setup: Five Layer Neural Network",
        "Convolution Neural Network",
        "Convolution Neural Network Inferencing with helper node",
        "Convolution Neural Network Split"
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
        elif choice == "Convolution Neural Network":
            window.destroy()
            CNN_layer.call()
        elif choice == "Convolution Neural Network Inferencing with helper node":
            window.destroy()
            CNN_helper.call()
        elif choice == "Convolution Neural Network Split":
            window.destroy()
            CNN_Split.call()
               
        

    clicked = StringVar()
    my_combo = ttk.Combobox(canvas_2, values=options,font=('sans 20 bold'), justify=CENTER, textvariable=clicked, style='W.TCombobox', state = "readonly")
    my_combo.grid(row = 0 , column=0, ipadx=200,ipady=10, padx=10, pady=10)
    # my_combo.config(dropdown_font = ('Times 20 bold'))
    my_combo.set( "Convolution Neural Network" )
    my_combo.bind("<<ComboboxSelected>>", display_selected)
    image_paths_uploaded = [
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image7.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png"
    ]
    num_columns_uploaded = len(image_paths_uploaded)
    #row_num = 2  # Start images from the second row
    available_width = WIDTH - 20  # Subtracting some padding
    available_height = 200  # Subtracting some padding and space for the combobox

    max_width_uploaded = (available_width // num_columns_uploaded)
    max_height_uploaded = available_height
    max_size_uploaded = min(available_width // num_columns_uploaded, available_height)

    uploaded_frame = Frame(window)
    uploaded_frame.pack(side=TOP, fill=X,pady=2)
    
    image_paths_additional = [
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image7.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png"
    ]
    image_paths_additional2 = [
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image7.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png"
    ]
    image_paths_additional3 = [
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image7.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png"
         
    ]
    
    image_paths_additional4 = [
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image7.png",
        "/home/ankit/Downloads/iudx-MOTION2NX/GUI/222.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/33333.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/abcd.png",
         "/home/ankit/Downloads/iudx-MOTION2NX/GUI/image6.png"
    ]
    
    
    

    available_width = WIDTH - 20  # Subtracting some padding
    available_height = 200  # Set a fixed height for the images
    num_columns_additional = len(image_paths_additional)
    
    uploaded_frame = Frame(window)
    uploaded_frame.pack(side=TOP, fill=X, pady=10)
    
   
    for i,image_path in enumerate(image_paths_uploaded):
        
        image = Image.open(image_path)
        image=image.resize((max_size_uploaded,max_size_uploaded), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(uploaded_frame, image=photo_image)
        label.image = photo_image  # Keep a reference to avoid garbage collection
        label.grid(row=0, column=i, padx=5, pady=5)
        image_radio_button = ImageRadioButton(uploaded_frame, image_path=image_path, value=i)
        # Bind the click event of the label to select the associated radio button
        label.bind("<Button-1>", lambda event, rb=image_radio_button: select_radio_button(rb))
    
    additional_frame = Frame(window)
    additional_frame.pack(side=TOP, fill=X, pady=10)
    
    for i, image_path in enumerate(image_paths_additional):
        image = Image.open(image_path)
        image = image.resize((max_size_uploaded,max_size_uploaded), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(additional_frame, image=photo_image)
        label.image = photo_image  # Keep a reference to avoid garbage collection
        label.grid(row=0, column=i, padx=5,pady=5)
        image_radio_button = ImageRadioButton(uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event, rb=image_radio_button: select_radio_button(rb))
    
    additional_frame2 = Frame(window)
    additional_frame2.pack(side=TOP, fill=X, pady=10)
    
    for i, image_path in enumerate(image_paths_additional2):
        image = Image.open(image_path)
        image = image.resize((max_size_uploaded,max_size_uploaded), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(additional_frame2, image=photo_image)
        label.image = photo_image  # Keep a reference to avoid garbage collection
        label.grid(row=0, column=i, padx=5,pady=5)
        image_radio_button = ImageRadioButton(uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event, rb=image_radio_button: select_radio_button(rb))
   
    additional_frame3 = Frame(window)
    additional_frame3.pack(side=TOP, fill=X, pady=10)
    
    for i, image_path in enumerate(image_paths_additional3):
        image = Image.open(image_path)
        image = image.resize((max_size_uploaded,max_size_uploaded), Image.ANTIALIAS)
        photo_image = ImageTk.PhotoImage(image)
        label = Label(additional_frame3, image=photo_image)
        label.image = photo_image  # Keep a reference to avoid garbage collection
        label.grid(row=0, column=i, padx=5,pady=5)
        image_radio_button = ImageRadioButton(uploaded_frame, image_path=image_path, value=i)
        label.bind("<Button-1>", lambda event, rb=image_radio_button: select_radio_button(rb))
    
    canvas_2 = Canvas(window, width=WIDTH, height=70)
    canvas_2.pack(side=TOP, fill=X)
        
    window.resizable(False,False)
    window.mainloop()
def select_radio_button(rb):
    rb.variable.set(rb.value)
    rb.on_select()





