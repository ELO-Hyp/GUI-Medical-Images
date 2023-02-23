import glob
import os
import threading
from functools import partial
from tkinter import Label, StringVar, filedialog
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import onnxruntime
import pydicom
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom


class RegistrationWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(700, 350)
        self.window.resizable(False, False)

        self.label_scale = Label(window, text="Select registration mode:", font='Arial 12')
        self.label_scale.place(relx=0.7, rely=0.05)
        self.model_name = StringVar(window, "2")
        iter = 0
        value_scale_dict = {"Arterial to Native": "model_art2nat", "Venous to Native": "model_ven2nat"}
        for (text, value) in value_scale_dict.items():
            iter += 0.1
            ttk.Radiobutton(window, text=text, variable=self.model_name,
                            value=value).place(relx=0.7, rely=0.05 + iter)

        # Folder input native folder.
        self.native_folder_imgs = Label(window, text="Processing folder with NATIVE scans:", fg="black", font='Arial 12')
        self.native_folder_imgs.place(relx=0.05, rely=0.05)
        self.native_folder_imgs_path = Label(window, text="", fg="black")
        self.native_folder_imgs_path.place(relx=0.05, rely=0.11)
        fct_folder_imgs = partial(self.select_folder, self.native_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse folder", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.18)

        # Folder input contrast folder.
        self.contrast_folder_imgs = Label(window, text="Processing folder with CONTRAST scans:", fg="black", font='Arial 12')
        self.contrast_folder_imgs.place(relx=0.05, rely=0.28)
        self.contrast_folder_imgs_path = Label(window, text="", fg="black")
        self.contrast_folder_imgs_path.place(relx=0.05, rely=0.34)
        fct_folder_imgs = partial(self.select_folder, self.contrast_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse folder", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.41)

        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.51)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.05, rely=0.57)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.64)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.42, rely=0.86)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.85)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__info_text = ""
        self.label_info = Label(window, text="", fg="black")
        self.label_info.place(relx=0.3, rely=0.93)
        self.__update()

        def on_closing():
            self.stop_thread = True
            self.window.grab_release()  # Release the main.
            self.window.destroy()

        # load network in memory
        self.scan_shape = (256, 512, 512)
        self.x_grid = np.linspace(0, self.scan_shape[0] - 1, self.scan_shape[0])
        self.y_grid = np.linspace(0, self.scan_shape[1] - 1, self.scan_shape[1])
        self.z_grid = np.linspace(0, self.scan_shape[2] - 1, self.scan_shape[2])

        self.model_art2nat = onnxruntime.InferenceSession(os.path.join("resources", "resources_gen", "registration_classic_art2nat.onnx"))
        self.model_ven2nat = onnxruntime.InferenceSession(os.path.join("resources", "resources_gen", "registration_classic_ven2nat.onnx"))

        self.window.iconbitmap(os.path.join("resources", 'elo-hyp_logo.ico'))
        self.window.protocol("WM_DELETE_WINDOW", on_closing)

    def select_folder(self, storing_label):
        filename = filedialog.askdirectory(title="Select a Folder")
        storing_label.configure(text=filename)

    def __update(self):
        if self.processing_thread is None:
            self.label_processing.configure(text=self.__shown_text)
        elif self.processing_thread.is_alive():
            self.label_processing.configure(text=f"Processing...")
        else:
            self.label_processing.configure(text=self.__shown_text)

        self.label_info.configure(text=self.__info_text)
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

        # pixels = pixels + intercept
        pixels = pixels / 1e3
        pixels = pixels - 1
        pixels = pixels.astype(np.float32)

        return dicom, pixels, padding_location, intercept

    def __inverse_CT_value(self, pixels):
        pixels = pixels + 1.0
        pixels = pixels * 1e3
        pixels[pixels < 0.0] = 0.0

        return pixels

    def read_scan(self, file_paths: list):
        slices = []
        for file_path in file_paths:
            if os.path.isdir(file_path) is True:
                continue
            try:
                dicom, pixels, padding_location, intercept = self.__read_CT(file_path)
                slices.append(pixels)
            except Exception as ex:
                print(ex)
                self.__shown_text = "An exception occurred!"

        return np.stack(slices)

    def run_registrantion(self, product_contrast: np.array, product_native: np.array, model_name: str):
        model = self.__getattribute__(model_name)

        # Data sanity check
        if product_contrast.shape[0] != product_native.shape[0]:
            raise Exception("Number of Native and Contrast slices is not equal!")
        no_slices = product_contrast.shape[0]
        if no_slices < self.scan_shape[0]:
            product_contrast = np.concatenate((product_contrast,
                                               np.zeros((self.scan_shape[0] - no_slices,
                                                         self.scan_shape[1],
                                                         self.scan_shape[2]))), 0)
            product_native = np.concatenate((product_native,
                                             np.zeros((self.scan_shape[0] - no_slices,
                                                       self.scan_shape[1],
                                                       self.scan_shape[2]))), 0)

        if no_slices > self.scan_shape[0]:
            product_contrast = product_contrast[:self.scan_shape[0]]
            product_native = product_native[:self.scan_shape[0]]

        # Downscale by 2 the input data
        product_contrast_ = zoom(product_contrast, 0.5)
        product_native_ = zoom(product_native, 0.5)

        input_prod = np.expand_dims(np.stack((product_contrast_, product_native_)), 0).astype(np.float32)
        ort_inputs = {model.get_inputs()[0].name: input_prod}
        ort_outs = model.run(None, ort_inputs)[0][0]

        # Define interpolator
        interpolator = RegularGridInterpolator((self.x_grid, self.y_grid, self.z_grid), product_contrast,
                                               bounds_error=False)
        output = interpolator(np.moveaxis(ort_outs, 0, -1))
        return output[:no_slices]

    def save_aligned_scan(self, output_dir, file_names, scan):
        for idx, file_name in enumerate(file_names):
            dicom = pydicom.dcmread(file_name)
            dicom.PixelData = np.array(scan[idx], np.int16).tobytes()

            out_path = os.path.join(output_dir, os.path.basename(file_name))
            dicom.save_as(out_path)

    def __process(self, input_dir_native: str, input_dir_contrast: str, output_dir: str, model_name: str):
        try:
            product_native = self.read_scan(glob.glob(os.path.join(input_dir_native, '*')))
            product_contrast = self.read_scan(glob.glob(os.path.join(input_dir_contrast, '*')))
            aligned_contrast = self.run_registrantion(product_contrast, product_native, model_name)
            aligned_contrast = self.__inverse_CT_value(aligned_contrast)

            self.save_aligned_scan(output_dir, glob.glob(os.path.join(input_dir_native, '*')), aligned_contrast)

            self.__shown_text = f"Results saved at: {output_dir}"
            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

    def __start_processing(self):

        input_dir_native = self.native_folder_imgs_path["text"]
        if input_dir_native is None or input_dir_native == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        input_dir_contrast = self.contrast_folder_imgs_path["text"]
        if input_dir_contrast is None or input_dir_contrast == "":
            messagebox.showerror("Error", "The processing folder must be selected!")
            return

        save_dir = self.label_saving_folder_path["text"]
        if save_dir is None or save_dir == "":
            messagebox.showerror("Error", "The saving folder must be selected!")
            return

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(input_dir_native, input_dir_contrast, save_dir,
                                                        str(self.model_name.get()), ), daemon=True)
        self.processing_thread.start()
