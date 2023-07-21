import os
import tkinter as tk
from tkinter import ttk

import psutil
from PIL import ImageTk, Image

from generation_window import GenerationWindow
from registration_contrast_window import RegistrationTransferWindow
from registration_window import RegistrationWindow
from seg_window import SegmentationWindow
from sr_window import SuperResolutionWindow


class MainWindow:
    def __init__(self, root, name):
        self.root = root
        self.root.title(name)

        # Set theme.
        self.root.tk.call("source", os.path.join("resources", "UI", "sun-valley.tcl"))
        self.root.tk.call("set_theme", "light")

        # Set the geometry of tkinter frame
        self.root.geometry("350x450")
        # Add image logo.
        image1 = Image.open(os.path.join("resources", "elo-hyp_logo.png")).resize((100, 100))
        test = ImageTk.PhotoImage(image1)

        label1 = tk.Label(image=test)
        label1.image = test
        # Position image
        label1.pack()

        ttk.Label(text="").pack()  # just for spacing.

        # Create button for the SR
        ttk.Button(self.root, text="Super-Resolution", command=self.__get_sr_window, width=30).pack()

        # Create button for the Segmentation
        ttk.Button(self.root, text="Abdominal Organ Segmentation", command=self.__get_seg_window, width=30).pack()

        ttk.Button(self.root, text="Contrast-Generation", command=self.__get_generation_window, width=30).pack()

        # Create buttons for alignment
        # Check memory constraint; minimum 26GB for alignment
        mem_usage = psutil.virtual_memory()
        total_memory = mem_usage.total/(1024 ** 3)
        disable_alignment = total_memory < 26

        button_alignment = ttk.Button(self.root, text="CT Alignment", command=self.__get_registration_window, width=30)
        button_alignment.pack()

        button_alignment_contrast = ttk.Button(self.root, text="CT Alignment with Contrast Transfer", width=30,
                                               command=self.__get_registration_transfer_window)
        button_alignment_contrast.pack()
        if disable_alignment:
            button_alignment.config(state='disabled')
            button_alignment_contrast.config(state='disabled')
            ttk.Label(text="*RAM requirement for CT alignment is 32 GB!").pack()

        ttk.Label(text="").pack()  # just for spacing.
        # Add norway logo.
        image_2 = Image.open(os.path.join("resources", "norway_grants_logo.png")).resize((50, 50))
        test = ImageTk.PhotoImage(image_2)

        label_2 = tk.Label(image=test)
        label_2.image = test
        # Position image
        label_2.pack()

        label_3 = ttk.Label(text="\n***The research leading to this application has received \nfunding from"
                                 " the NO Grants 2014-2021, under project\nELO-Hyp contract no. 24/2020.")
        label_3.pack()
        self.root.iconbitmap(os.path.join("resources", 'elo-hyp_logo.ico'))
        self.root.mainloop()




    def __get_sr_window(self):
        """Create a new top level window"""
        self.super_resolution_window = SuperResolutionWindow(tk.Toplevel(), "Super-Resolution")

    def __get_seg_window(self):
        """Create a new top level window"""
        self.segmentation_window = SegmentationWindow(tk.Toplevel(), "Abdominal Organ Segmentation")

    def __get_generation_window(self):
        """Create a new top level window"""
        self.contrast_generation_window = GenerationWindow(tk.Toplevel(), "Contrast-Generation")

    def __get_registration_window(self):
        """Create a new top level window"""
        self.contrast_generation_window = RegistrationWindow(tk.Toplevel(), "Registration")

    def __get_registration_transfer_window(self):
        """Create a new top level window"""
        self.contrast_generation_with_transfer_window = RegistrationTransferWindow(tk.Toplevel(), "Registration-Transfer")
