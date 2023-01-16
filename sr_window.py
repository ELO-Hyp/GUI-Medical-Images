from tkinter import Tk, Label, Button, StringVar, Radiobutton, filedialog
from tkinter import ttk
from tkinter import messagebox
import threading
import time
import onnxruntime
import numpy as np
import pickle
from functools import partial
import glob
import os
import pydicom
import pdb
import matplotlib.pyplot as plt


class SuperResolutionWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(1000, 350)
        self.window.resizable(False, False)

        # Select SR scale.

        self.label_scale = Label(window, text="Select resolution scale:", font='Arial 12')
        self.label_scale.place(relx=0.05, rely=0.05)
        self.resolution_scale = StringVar(window, "2")
        iter = 0
        value_scale_dict = {"X2": 2, "X4": 4}
        for (text, value) in value_scale_dict.items():
            iter += 0.05
            Radiobutton(window, text=text, variable=self.resolution_scale,
                        value=value).place(relx=0.2 + iter, rely=0.05)

        # Select dicom type.
        self.label_type = Label(window, text="Select image type:", font='Arial 12')
        self.label_type.place(relx=0.05, rely=0.15)
        self.img_type = StringVar(window, "CT")
        iter = 0
        value_scale_dict = {"CT": "CT", "MRI": "MRI"}
        for (text, value) in value_scale_dict.items():
            iter += 0.05
            Radiobutton(window, text=text, variable=self.img_type,
                        value=value).place(relx=0.2 + iter, rely=0.15)

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing folder with scans:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.25)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.3, rely=0.25)
        fct_folder_imgs = partial(self.select_folder, self.label_folder_imgs_path)
        self.button_folder = Button(window, text="Browse folder", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.32)

        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.4)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.3, rely=0.4)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.47)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.42, rely=0.86)

        self.button_start_processing = Button(window, text="Start process", font='Arial 11',
                                                      command=self.__start_processing)
        self.button_start_processing.place(relx=0.3, rely=0.85)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()

        def on_closing():
            self.stop_thread = True
            self.window.grab_release() # Release the main.
            self.window.destroy()

        # load network in memory
        self.model_ct_x2 = onnxruntime.InferenceSession(os.path.join("resources_sr", "CT_x2_single.onnx"))
        self.model_ct_x4 = onnxruntime.InferenceSession(os.path.join("resources_sr", "CT_x4_single.onnx"))
        self.model_mri_x2 = onnxruntime.InferenceSession("super_resolution.onnx")
        self.model_mri_x4 = onnxruntime.InferenceSession("super_resolution.onnx")

        self.window.protocol("WM_DELETE_WINDOW", on_closing)

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing: {self.counter}/{self.__num_of_processing_images}.")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.window.after(1000, self.__update)

    def __run_network(self, model, img):
        lr = np.expand_dims(img, axis=(0, 1))

        lr = np.float32(lr)
        # compute ONNX Runtime output prediction
        ort_inputs = {model.get_inputs()[0].name: lr}
        ort_outs = model.run(None, ort_inputs)

        return ort_outs[0][0, 0]

    def __read_CT(self, dicom_path: str):
        dicom = pydicom.dcmread(dicom_path)
        intercept = int(dicom[0x28, 0x1052].value)
        pixels = dicom.pixel_array
        padding = -2000
        padding_location = np.where(pixels == padding)
        pixels[pixels == padding] = 0

        pixels = pixels + intercept
        pixels = pixels / 1000
        return dicom, pixels, padding_location, intercept

    def __inverse_CT_value(self, pixels, intercept, scale, padding_location):
        pixels = pixels * 1000
        pixels = pixels - intercept

        return pixels

    def run_sr(self, file_path: str, scale: int, img_type: str):
        try:
            if img_type == 'CT':
                dicom, pixels, padding_location, intercept = self.__read_CT(file_path)

                if scale == 2:
                    sr_pixels = self.__run_network(self.model_ct_x2, pixels)
                elif scale == 4:
                    sr_pixels = self.__run_network(self.model_ct_x4, pixels)
                else:
                    raise ValueError(f"Scale {scale} not supported!")
                sr_pixels = self.__inverse_CT_value(sr_pixels, intercept=intercept, scale=scale,
                                                    padding_location=padding_location)
                # plt.subplot(1, 2, 1)
                # plt.imshow(pixels, cmap='gray')
                # plt.subplot(1, 2, 2)
                # plt.imshow(sr_pixels, cmap='gray')
                # plt.show()
                dicom.PixelData = np.array(sr_pixels, np.int16).tobytes()
                dicom.Rows = int(dicom.Rows * int(scale))
                dicom.Columns = int(dicom.Columns * int(scale))
            elif img_type == 'MRI':
                pass
            else:
                raise ValueError(f"Image type {img_type} not supported!")
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

        return dicom

    def __process(self, input_dir: str, output_dir: str, scale: int, img_type: str):
        try:
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            self.__num_of_processing_images = len(file_paths)
            for file_path in file_paths:
                dicom = self.run_sr(file_path, scale=scale, img_type=img_type)
                path_to_save = os.path.join(output_dir, os.path.split(file_path)[-1])
                dicom.save_as(path_to_save)
                self.counter += 1

            self.__shown_text = f"Results saved at: {output_dir}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):

        input_dir = self.label_folder_imgs_path["text"]
        if input_dir is None or input_dir == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir, save_dir,
                                                        int(self.resolution_scale.get()),
                                                        self.img_type.get(), ), daemon=True)
        self.processing_thread.start()