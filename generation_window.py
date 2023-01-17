import glob
import os
import threading
from functools import partial
from tkinter import Label, Button, StringVar, Radiobutton, filedialog
from tkinter import messagebox

import numpy as np
import onnxruntime
import pydicom


class GenerationWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(1000, 350)
        self.window.resizable(False, False)

        # Select SR scale.

        self.label_scale = Label(window, text="Select generation mode:", font='Arial 12')
        self.label_scale.place(relx=0.5, rely=0.05)
        self.gen_model = StringVar(window, "2")
        iter = 0
        value_scale_dict = {"Arterial to Native": "model_art2nat", "Native to Arterial": "model_nat2art",
                            "Venous to Native": "model_ven2nat", "Native to Venous": "model_nat2ven"}
        for (text, value) in value_scale_dict.items():
            iter += 0.1
            Radiobutton(window, text=text, variable=self.gen_model,
                        value=value).place(relx=0.5, rely=0.05 + iter)

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing folder with scans:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.05, rely=0.20)
        fct_folder_imgs = partial(self.select_folder, self.label_folder_imgs_path)
        self.button_folder = Button(window, text="Browse folder", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.12)

        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.25)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.05, rely=0.42)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.33)

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
        self.model_art2nat = onnxruntime.InferenceSession(os.path.join("resources_gen", "art2nat.onnx"))
        self.model_nat2art = onnxruntime.InferenceSession(os.path.join("resources_gen", "art2nat.onnx"))
        self.model_ven2nat = onnxruntime.InferenceSession(os.path.join("resources_gen", "art2nat.onnx"))
        self.model_nat2ven = onnxruntime.InferenceSession(os.path.join("resources_gen", "art2nat.onnx"))

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

        # pixels = pixels + intercept
        pixels = pixels / 1e3
        pixels = pixels - 1
        pixels = pixels.astype(np.float32)

        return dicom, pixels, padding_location, intercept

    def __inverse_CT_value(self, pixels, intercept, padding_location):
        pixels = pixels + 1.0
        pixels = pixels * 1e3
        pixels[pixels < 0.0] = 0.0

        return pixels

    def run_gen(self, file_path: str, gen_model: str):
        try:
            dicom, pixels, padding_location, intercept = self.__read_CT(file_path)

            model = self.__getattribute__(gen_model)
            output = self.__run_network(model, pixels)

            sr_pixels = self.__inverse_CT_value(output, intercept=intercept, padding_location=padding_location)
            # plt.subplot(1, 2, 1)
            # plt.imshow(pixels, cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(sr_pixels, cmap='gray')
            # plt.show()
            dicom.PixelData = np.array(sr_pixels, np.int16).tobytes()
            dicom.Rows = int(dicom.Rows)
            dicom.Columns = int(dicom.Columns)
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"

        return dicom

    def __process(self, input_dir: str, output_dir: str, gen_model: str):
        try:
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            self.__num_of_processing_images = len(file_paths)
            for file_path in file_paths:
                dicom = self.run_gen(file_path, gen_model=gen_model)
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
                                                        str(self.gen_model.get()), ), daemon=True)
        self.processing_thread.start()
