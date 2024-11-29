from tkinter import *
from tkinter import ttk
from utility import paint
from utility import File_location
import time
from utility import SMPC
from utility import NN
from utility import NN_helper
from utility import two_layer
from utility import CNN_layer
from utility import CNN_helper
from utility import CNN_Split
from utility import four_layer
from utility import five_layer
from utility import Six_layer

# window = Tk()


def call():
    window = Tk()
    window.title("SMPC")

    WIDTH = 600
    HEIGHT = 575

    window.geometry("1200x1040")

    canvas_2 = Canvas(window, width=WIDTH, height=70)
    canvas_2.grid(row=0, column=0)
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
        #   "Convolution Neural Network",
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
      #  elif choice == "Convolution Neural Network":
       #     window.destroy()
        #    CNN_layer.call()
        elif choice == "Convolution Neural Network Inferencing with helper node":
            window.destroy()
            CNN_helper.call()
        elif choice == "Convolution Neural Network Split":
            window.destroy()
            CNN_Split.call()
        elif choice == "Setup: Four Layer Convolution Network":
            window.destroy()
            four_layer.call()
        elif choice == "Setup: Five Layer Neural Network":
            window.destroy()
            five_layer.call()
        elif choice == "Setup: Six Layer Convolution Network":
            window.destroy()
            Six_layer.call()

    clicked = StringVar()
    my_combo = ttk.Combobox(canvas_2, values=options, font=(
        'sans 20 bold'), justify=CENTER, textvariable=clicked, style='W.TCombobox', state="readonly")
    my_combo.grid(row=0, column=0, ipadx=280, ipady=10, padx=10, pady=10)
    # my_combo.config(dropdown_font = ('Times 20 bold'))
    my_combo.set("CNN 4 Layer")
    my_combo.bind("<<ComboboxSelected>>", display_selected)

    # canvas_1 = Canvas(window, width= WIDTH, height=HEIGHT)
    canvas_1 = Canvas(window, width=1000, height=250)
    canvas_1.grid(row=1, column=0, padx=20)

    # back_image2 = PhotoImage(file="./Images_Video/NN.png")
    # my_image2 = canvas_1.create_image(WIDTH/2,HEIGHT/2,anchor=CENTER,image = back_image2)

    text_1 = "Configuration"
    canvas_1.create_text(20, 25, anchor=W, text=text_1,
                         fill="black", font=('sans 20 bold'))
    text_30 = "Layer"
    text_31 = "No. of kernels"
    text_32 = "No. of Mult"
    text_33 = "No. of Parameters"
    canvas_1.create_text(20, 60, anchor=W, text=text_30,
                         fill="black", font=('sans 18 bold'))
    canvas_1.create_text(300, 60, anchor=W, text=text_31,
                         fill="black", font=('sans 18 bold'))
    canvas_1.create_text(560, 60, anchor=W, text=text_32,
                         fill="black", font=('sans 18 bold'))
    canvas_1.create_text(755, 60, anchor=W, text=text_33,
                         fill="black", font=('sans 18 bold'))

    text_40 = "CNN1"
    text_41 = "5"
    text_42 = "21,125"
    text_43 = "5 x 1 X 5 X 5"
    canvas_1.create_text(20,  100, anchor=W, text=text_40,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(340, 100, anchor=W, text=text_41,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(600, 100, anchor=W, text=text_42,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(820, 100, anchor=W, text=text_43,
                         fill="black", font=('sans 16 '))
    text_50 = "CNN2"
    text_51 = "3"
    text_52 = "8640"
    text_53 = "3 x 5 X 4 X 4"

    canvas_1.create_text(20, 140, anchor=W, text=text_50,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(340, 140, anchor=W, text=text_51,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(600, 140, anchor=W, text=text_52,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(820, 140, anchor=W, text=text_53,
                         fill="black", font=('sans 16 '))
    text_60 = "NN3"
    text_61 = "-"
    text_62 = "10,800"
    text_63 = "108 X 100"
    canvas_1.create_text(20, 180, anchor=W, text=text_60,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(340, 180, anchor=W, text=text_61,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(600, 180, anchor=W, text=text_62,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(820, 180, anchor=W, text=text_63,
                         fill="black", font=('sans 16 '))
    text_70 = "NN4"
    text_71 = "-"
    text_72 = "1000"
    text_73 = "100 X 10"
    canvas_1.create_text(20, 220, anchor=W, text=text_70,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(340, 220, anchor=W, text=text_71,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(600, 220, anchor=W, text=text_72,
                         fill="black", font=('sans 16 '))
    canvas_1.create_text(820, 220, anchor=W, text=text_73,
                         fill="black", font=('sans 16 '))
    canvas_2 = Canvas(window, width=1000, height=230)
    canvas_2.grid(row=2, column=0, padx=20)
    TEXT_1 = "System Configuration (WAN)"
    canvas_2.create_text(20, 25, anchor=W, text=TEXT_1,
                         fill="black", font=('sans 20 bold'))
    TEXT_30 = "Server 0"
    TEXT_31 = "0.8 GB RAM"
    TEXT_32 = "2.6 GHZ"
    canvas_2.create_text(20, 60, anchor=W, text=TEXT_30,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(340, 60, anchor=W, text=TEXT_31,
                         fill="black", font=('sans 16'))
    canvas_2.create_text(600, 60, anchor=W, text=TEXT_32,
                         fill="black", font=('sans 16'))
    TEXT_40 = "Server 1"
    TEXT_41 = "0.8 GB RAM"
    TEXT_42 = "2.4 GHZ"
    canvas_2.create_text(20, 100, anchor=W, text=TEXT_40,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(340, 100, anchor=W, text=TEXT_41,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(600, 100, anchor=W, text=TEXT_42,
                         fill="black", font=('sans 16 '))
    TEXT_50 = "Server 2"
    TEXT_51 = "0.8 GB RAM"
    TEXT_52 = "2.4 GHZ"
    canvas_2.create_text(20, 140, anchor=W, text=TEXT_50,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(340, 140, anchor=W, text=TEXT_51,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(600, 140, anchor=W, text=TEXT_52,
                         fill="black", font=('sans 16 '))
    TEXT_60 = "Local system Configuration"
    canvas_2.create_text(20, 180, anchor=W, text=TEXT_60,
                         fill="black", font=('sans 18 bold '))
    TEXT_60 = "No. of cores: 4"
    TEXT_61 = "16 GB RAM"
    TEXT_62 = "2.5 GHZ"
    canvas_2.create_text(20, 220, anchor=W, text=TEXT_60,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(340, 220, anchor=W, text=TEXT_61,
                         fill="black", font=('sans 16 '))
    canvas_2.create_text(600, 220, anchor=W, text=TEXT_62,
                         fill="black", font=('sans 16 '))

    canvas_3 = Canvas(window, width=1100, height=550)
    canvas_3.grid(row=3, column=0, padx=20, pady=20)
    text_90 = "SPLIT Configuration"
    Label_90 = Label(canvas_3, text=text_90, font=(
        'sans 18 bold'), height=2, width=20)
    Label_90.grid(row=0, column=0, columnspan=3, sticky='w')
    text_00 = "Configuration"
    text_10 = "RAM(GB)\nLAN"
    text2_10 = "RAM(GB)\nWAN"
    text_20 = "Time(Sec)\nLAN"
    text2_20 = "Time(Sec)\nWAN"
    text1_30 = "No. of Mult Per Split"
    Label_00 = Label(canvas_3, text=text_00, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=16)
    Label_00.grid(row=1, column=0)
    Label_10 = Label(canvas_3, text=text_10, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_10.grid(row=1, column=1)
    Label_20 = Label(canvas_3, text=text2_10, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_20.grid(row=1, column=2)
    Label_30 = Label(canvas_3, text=text_20, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_30.grid(row=1, column=3)
    Label_40 = Label(canvas_3, text=text2_20, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_40.grid(row=1, column=4)
    Label_40 = Label(canvas_3, text=text1_30, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=24)
    Label_40.grid(row=1, column=5)

    text_01 = "CNN_split:(1,1)\nNN_split:(1,1)"
    text_11 = "0.219"
    text_21 = "0.229"
    text1_41 = "17"
    text1_51 = "21"
    text1_31 = "21,125 | 8640 | 10,800 | 1060"
    Label_01 = Label(canvas_3, text=text_01, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=16)
    Label_01.grid(row=2, column=0)
    Label_11 = Label(canvas_3, text=text_11, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_11.grid(row=2, column=1)
    Label_21 = Label(canvas_3, text=text_21, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_21.grid(row=2, column=2)
    Label_31 = Label(canvas_3, text=text1_41, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_31.grid(row=2, column=3)
    Label_41 = Label(canvas_3, text=text1_51, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_41.grid(row=2, column=4)
    Label_51 = Label(canvas_3, text=text1_31, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=24)
    Label_51.grid(row=2, column=5)

    text_02 = "CNN_split:(5,3)\nNN_split:(1,1)"
    text_12 = "0.219"
    text_22 = "0.228"
    text1_42 = "30"
    text1_52 = "33"
    text1_32 = "4265 | 2800 | 10,800 | 1000"
    Label_02 = Label(canvas_3, text=text_02, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=16)
    Label_02.grid(row=3, column=0)
    Label_12 = Label(canvas_3, text=text_12, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_12.grid(row=3, column=1)
    Label_22 = Label(canvas_3, text=text_22, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_22.grid(row=3, column=2)
    Label_32 = Label(canvas_3, text=text1_42, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_32.grid(row=3, column=3)
    Label_42 = Label(canvas_3, text=text1_52, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_42.grid(row=3, column=4)
    Label_52 = Label(canvas_3, text=text1_32, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=24)
    Label_52.grid(row=3, column=5)

    text_03 = "CNN_split:(1,1)\nNN_split:(20,2)"
    text_13 = "0.109"
    text_23 = "0.089"
    text1_43 = "31"
    text1_53 = "42"
    text1_33 = "21,125 | 8640 | 540 | 500"
    Label_03 = Label(canvas_3, text=text_03, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=16)
    Label_03.grid(row=4, column=0)
    Label_13 = Label(canvas_3, text=text_13, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_13.grid(row=4, column=1)
    Label_23 = Label(canvas_3, text=text_23, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_23.grid(row=4, column=2)
    Label_33 = Label(canvas_3, text=text1_43, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_33.grid(row=4, column=3)
    Label_43 = Label(canvas_3, text=text1_53, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_43.grid(row=4, column=4)
    Label_53 = Label(canvas_3, text=text1_33, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=24)
    Label_53.grid(row=4, column=5)

    text_04 = "CNN_split:(5,3)\nNN_split:(20,2)"
    text_14 = "0.035"
    text_24 = "0.036"
    text1_44 = "43"
    text1_54 = "54"
    text1_34 = "4265 | 2800 | 540 | 500"
    Label_04 = Label(canvas_3, text=text_04, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=16)
    Label_04.grid(row=5, column=0)
    Label_14 = Label(canvas_3, text=text_14, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_14.grid(row=5, column=1)
    Label_24 = Label(canvas_3, text=text_24, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_24.grid(row=5, column=2)
    Label_34 = Label(canvas_3, text=text1_44, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_34.grid(row=5, column=3)
    Label_44 = Label(canvas_3, text=text1_54, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_44.grid(row=5, column=4)
    Label_44 = Label(canvas_3, text=text1_34, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=24)
    Label_44.grid(row=5, column=5)

    text_05 = "Helper Node"
    text_15 = "0.042"
    text_25 = "0.04"
    text1_45 = "14"
    text1_55 = "15"
    text1_35 = "21,125 | 8640 | 10,800 | 1060"
    Label_05 = Label(canvas_3, text=text_05, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=16)
    Label_05.grid(row=6, column=0)
    Label_15 = Label(canvas_3, text=text_15, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_15.grid(row=6, column=1)
    Label_25 = Label(canvas_3, text=text_25, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_25.grid(row=6, column=2)
    Label_35 = Label(canvas_3, text=text1_45, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_35.grid(row=6, column=3)
    Label_35 = Label(canvas_3, text=text1_55, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=10)
    Label_35.grid(row=6, column=4)
    Label_35 = Label(canvas_3, text=text1_35, highlightthickness=1,
                     highlightbackground="black", font=('sans 16 normal'), height=2, width=24)
    Label_35.grid(row=6, column=5)

    # text_06 = "H3:(1,2,-,-)"
    # text_16 = "0.035"
    # text_26 = "60"
    # text1_36= "4265, 1440, 540, 500"
    # Label_06 = Label(canvas_2, text = text_06,highlightthickness=1, highlightbackground="black",font=('sans 16 normal'), height=2, width=20)
    # Label_06.grid(row = 7, column=0)
    # Label_16 = Label(canvas_2, text = text_16,highlightthickness=1, highlightbackground="black",font=('sans 16 normal'), height=2, width=20)
    # Label_16.grid(row = 7, column=1)
    # Label_26 = Label(canvas_2, text = text_26,highlightthickness=1, highlightbackground="black",font=('sans 16 normal'), height=2, width=20)
    # Label_26.grid(row = 7, column=2)
    # Label_36 = Label(canvas_2, text = text1_36,highlightthickness=1, highlightbackground="black",font=('sans 16 normal'), height=2, width=20)
    # Label_36.grid(row = 7, column=3)

    window.resizable(False, False)
    window.mainloop()