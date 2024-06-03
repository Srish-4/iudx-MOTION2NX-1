from tkinter import *
from tkinter import filedialog, PhotoImage
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfile
from utility import SMPC
from utility import NN
from utility import loading
from utility import CNN_layer
import subprocess
import os


def call(File_location2):

    try:
        root = Tk()
        global myLabel, filepath
        filepath = File_location2 
        root.title("Upload File")
        # root.attributes('-fullscreen', True)


        root.geometry("1080x800")

        myLabel = Label(root)
        
        
        def FilePath(selected_value):
            
          folder_number = selected_value  # Assuming selected_value is the user input
          folder_path = f"/home/ankit/Downloads/iudx-MOTION2NX/data/ImageProvider/sample_data/images_folder{folder_number}"
          selected_files = filedialog.askopenfilenames(initialdir=folder_path, title="Select CSV files", filetypes=[("CSV files", "*.csv")])
          print("Selected files:", selected_files)
        
        def back(root):
            root.destroy()
            NN.call()
            
            

        myButton = Button(root, text= "Upload File", command=lambda: filepath(0), width=15)
        myButton = Button(root, text= "Upload File", command= FilePath, width=15)
        myButton.grid(row=1,column=1, padx=5,pady=10)
        # myButton.pack()

        showButton = Button(root, text= "Show Image", command= lambda: openFile(filepath), width=15)
        showButton.grid(row=3,column=0, columnspan=2, padx=75,pady=10)
        # showButton.pack()


        nextButton = Button(root, text= "Send Seceret Shares", command= uplaod, width=30)
        nextButton.grid(row=4,column=0, columnspan=2, padx=75,pady=10)

        App_title = Label(root, text= "Upload your Image", command= None,font=('Times 20 bold'))
        App_title.grid(row=0,column=0,columnspan=2, padx=10,pady=10)

        back_Image = PhotoImage(file="./utility/Images_Video/back.png")
        backButton = Button(root, image = back_Image, command=lambda: back(root), width=50, height=50)
        backButton.grid(row=0,column=0,columnspan=2, padx=10,pady=10, sticky=W)

        root.resizable(False , False)

        root.mainloop()
    except EXCEPTION as e:
        # print(e)
        call(File_location2)
