from tkinter import Tk, Label, StringVar, filedialog
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
from pathlib import Path
import cv2 as cv


class SegmentationWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set()  # Make it the main window.

        self.review_enroll_window = None
        self.settings_window_created = False

        self.window.title(window_title)
        self.window.minsize(700, 150)
        self.window.resizable(False, False)

        # Folder input folder.
        self.label_folder_imgs = Label(window, text="Processing folder with scans:", fg="black", font='Arial 12')
        self.label_folder_imgs.place(relx=0.05, rely=0.05)
        self.label_folder_imgs_path = Label(window, text="", fg="black")
        self.label_folder_imgs_path.place(relx=0.35, rely=0.06)
        fct_folder_imgs = partial(self.select_folder, self.label_folder_imgs_path)
        self.button_folder = ttk.Button(window, text="Browse folder", command=fct_folder_imgs)
        self.button_folder.place(relx=0.05, rely=0.2)

        # Folder output folder.
        self.label_saving_folder = Label(window, text="Saving folder:", font='Arial 12')
        self.label_saving_folder.place(relx=0.05, rely=0.38)
        self.label_saving_folder_path = Label(window, text="")
        self.label_saving_folder_path.place(relx=0.35, rely=0.38)
        fct_folder_save = partial(self.select_folder, self.label_saving_folder_path)
        self.button_saving_folder = ttk.Button(window, text="Browse folder", command=fct_folder_save)
        self.button_saving_folder.place(relx=0.05, rely=0.53)

        self.counter = 0
        self.processing_thread = None

        self.label_processing = Label(window, text="",  fg="black")
        self.label_processing.place(relx=0.35, rely=0.7)

        self.button_start_processing = ttk.Button(window, text="Start process",
                                                  command=self.__start_processing)
        self.button_start_processing.place(relx=0.05, rely=0.75)
        self.stop_thread = False
        self.__shown_text = ""
        self.__num_of_processing_images = 0
        self.__update()

        def on_closing():
            self.stop_thread = True
            self.window.grab_release()  # Release the main.
            self.window.destroy()

        # load network in memory
        self.model_seg_resnet = onnxruntime.InferenceSession(os.path.join("resources",
                                                                          "resources_sr", "segmentation_resnet.onnx"))

        self.window.iconbitmap(os.path.join("resources", 'elo-hyp_logo.ico'))
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

    def get_img_from_pixels(self, pixels):

        if pixels.shape[0] > 320:
            top = int(np.ceil((pixels.shape[0] - 320) / 2))
            bottom = pixels.shape[0] - int(np.floor((pixels.shape[0] - 320) / 2))
            pixels = pixels[top:bottom, :]

        if pixels.shape[1] > 384:
            left = int(np.ceil((pixels.shape[1] - 384) / 2))
            right = pixels.shape[1] - int(np.floor((pixels.shape[1] - 384) / 2))
            pixels = pixels[:, left:right]

        shape0 = np.array(pixels.shape[:2])
        resize = np.array([320, 384])
        if np.any(shape0 != resize):
            diff = resize - shape0
            pad0 = diff[0]
            pad1 = diff[1]
            pady = [pad0 // 2, pad0 // 2 + pad0 % 2]
            padx = [pad1 // 2, pad1 // 2 + pad1 % 2]
            pixels = np.pad(pixels, [pady, padx])
            pixels = pixels.reshape((resize[0], resize[1]))

        img = np.float32(pixels) / pixels.max()
        img = np.dstack((img, img, img))
        img = img.transpose((2, 0, 1))
        return img

    def __run_network(self, model, img):

        lr = np.expand_dims(img, axis=(0))

        lr = np.float32(lr)
        # compute ONNX Runtime output prediction
        ort_inputs = {model.get_inputs()[0].name: lr}
        ort_outs = model.run(None, ort_inputs)
        seg = (ort_outs[0][0].transpose((1, 2, 0)) > 0) * 255
        return seg

    def __read_CT(self, dicom_path: str):
        dicom = pydicom.dcmread(dicom_path)
        pixels = dicom.pixel_array
        return pixels

    def __read_MRI(self, dicom_path: str):
        dicom = pydicom.dcmread(dicom_path)
        pixels = dicom.pixel_array
        return dicom, pixels

    def __inverse_CT_value(self, pixels, intercept, scale, padding_location):
        pixels = pixels * 1000
        pixels = pixels - intercept

        return pixels

    def run_seg(self, file_path: str):
        try:

            pixels = self.__read_CT(file_path)

            pixels = self.get_img_from_pixels(pixels)
            segmentation = self.__run_network(self.model_seg_resnet, pixels)
            pixels = np.transpose(pixels, (1, 2, 0))
            segmentation = np.float32(segmentation) / 255.0
            segmentation = cv.addWeighted(pixels, 0.5, segmentation, 0.5, 0)
            segmentation = np.uint8(segmentation * 255)

            segmentation = np.concatenate((np.zeros((segmentation.shape[0], 100, 3)), segmentation), axis=1)

            cv.rectangle(segmentation, (0, 0), (120, 80), (255, 255, 255), -1)
            cv.rectangle(segmentation, (5, 5), (15, 15), (255, 0, 0), -1)
            cv.putText(segmentation, 'Stomach', (16, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv.rectangle(segmentation, (5, 30), (15, 40), (0, 0, 255), -1)
            cv.putText(segmentation, 'Large bowel', (16, 41), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv.rectangle(segmentation, (5, 55), (15, 65), (0, 255, 0), -1)
            cv.putText(segmentation, 'Small bowel', (16, 66), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # plt.subplot(1, 2, 1)
            # plt.imshow(pixels, cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(segmentation)
            # plt.show()

            return segmentation
        except Exception as ex:
            print(file_path, ex)
            self.__shown_text = "An exception occurred!"

    def __process(self, input_dir: str, output_dir: str):
        try:
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            self.__num_of_processing_images = len(file_paths)
            for file_path in file_paths:
                try:
                    if os.path.isdir(file_path):
                        continue
                    segmentation_as_img = self.run_seg(file_path)
                    path_to_save = os.path.join(output_dir, Path(Path(file_path).parts[-1]).with_suffix(".png"))
                    cv.imwrite(path_to_save, segmentation_as_img)
                    self.counter += 1

                except Exception as ex:
                    print(ex)
                    self.__shown_text = "An exception occurred!"

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
                                                  args=(input_dir, save_dir, ), daemon=True)
        self.processing_thread.start()