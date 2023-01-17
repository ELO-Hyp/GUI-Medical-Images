import tkinter as tk
from tkinter import ttk
from sr_window import SuperResolutionWindow
from seg_window import SegmentationWindow

class MainWindow:
    def __init__(self, root, name):
        self.root = root

        # Set the geometry of tkinter frame
        self.root.geometry("750x250")

        # Create button for the SR
        ttk.Button(self.root, text="Super-Resolution", command=self.__get_sr_window).pack()

        # Create button for the Segmentation
        ttk.Button(self.root, text="Abdominal Organ Segmentation", command=self.__get_seg_window).pack()

        self.root.mainloop()


    def __get_sr_window(self):
        """Create a new top level window"""
        self.super_resolution_window = SuperResolutionWindow(tk.Toplevel(), "Super-Resolution")

    def __get_seg_window(self):
        """Create a new top level window"""
        self.segmentation_window = SegmentationWindow(tk.Toplevel(), "Abdominal Organ Segmentation")
