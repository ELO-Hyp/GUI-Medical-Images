import multiprocessing
from main_window import MainWindow
import tkinter

if __name__ == "__main__":
    multiprocessing.freeze_support()
    MainWindow(tkinter.Tk(), "ELO-HYP Medical Apps")
    
    
# pyinstaller app.py --onefile  -F --hiddenimport=pydicom.encoders.gdcm --hiddenimport=pydicom.encoders.pylibjpeg 