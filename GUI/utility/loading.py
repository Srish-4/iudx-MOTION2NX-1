from time import sleep
from random import random
from threading import Thread
 
from utility import File_location
from tkinter import *
from tkinter import ttk
import time
from tkinter import *
from tkVideoPlayer import TkinterVideo
from utility import result_Cifar10
from utility import result_mnist
import subprocess
import os
global options
global option
def choose_dataset(value):
  #print("$$$ I entered into user choice $$$$")
  global options
  options=value
  #print("loading.py 20 option value set to :", option)

def get_user_choice(value):
  #print("$$$ I entered into user choice $$$$")
  global option
  option=value
  #print("loading.py 20 option value set to :", option)

def call(image):
    # image = "Daksh.jpg"

    global my_progress_bar,videoplayer,root, temp

    root = Tk()
    root.title("Output share Receiver")
    root.geometry("300x300")

    # root.title("Application")

    # root.geometry("300x300")
    root.configure(bg='white')

    temp = True

    # long-running background task


    def background_task():
    	
        global videoplayer,temp,option
        #print("in loading.py 44 Value of option:",option)
        while True:
            if (not temp):
                break

            else:
                base_dir = os.getenv("BASE_DIR")
                f = open("output.txt", "w")
                if options == 1:
                  iter = subprocess.call("python3 " + base_dir + "/Dataprovider/image_provider/MNIST_preprocess_image.py -f "+image, shell=True)
                #subprocess.call(base_dir + "/scripts/ServerModel_Architecture/Split/ImageProvider.sh", stdout=f)
                elif options ==2:
                  iter = subprocess.call("python3 " + base_dir + "/Dataprovider/image_provider/cifar_preprocess_image.py -f "+image, shell=True)
                if option == 1:
                  #print(" option 1 is executed")
                  subprocess.call(base_dir + "/scripts/ServerModel_Architecture/HelperNode/ImageProvider.sh", stdout=f)
                elif option == 2:
                 #print(" option 2 is executed")
                 #print("**************************************")
                 subprocess.call(base_dir + "/scripts/ServerModel_Architecture/HelperNode/CNN/ImageProvider.sh", stdout=f)
                else:
                  print("Invalid choice.")
                temp = False
                time.sleep(1)


    # create and start the daemon thread
    print('Starting background task...')
    daemon = Thread(target=background_task, daemon=True, name='Monitor')
    daemon.start()
    # main thread is carrying on...
    print('Main thread is carrying on...')

    videoplayer = TkinterVideo(master= root, scaled = False)
    videoplayer.load("./utility/Images_Video/buffer1.mp4")

    videoplayer.pack(anchor=S, expand= True, fill= "both")
    
    # videoplayer.place(x = 100, y= 200)


    def loopVideo(event):
        global temp,my_progress_bar,videoplayer,root
        
        # print(temp)
        if temp:
            videoplayer.play()


        if not temp:
            if options==1:
              root.destroy()
              result_mnist.call(image)
            elif options ==2:
              root.destroy()
              result_Cifar10.call(image)


    videoplayer.play()

    temp = 3
    videoplayer.bind('<<Ended>>', loopVideo)


    root.mainloop()
    print('Main thread done.')

