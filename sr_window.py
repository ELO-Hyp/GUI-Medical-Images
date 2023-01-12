from tkinter import Tk, Label, Button, StringVar, Radiobutton, filedialog
from tkinter import ttk
from tkinter import messagebox
import threading
import time
import onnxruntime
import numpy as np
import pickle
from functools import partial


class SuperResolutionWindow:
    def __init__(self, window, window_title):

        self.window = window
        self.window.grab_set() # Make it the main window.

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

    def __test_onnx(self):
        ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

        # def to_numpy(tensor):
        #
        #     return tensor.cpu().numpy() # tensor.detach().cpu().numpy() if tensor.requires_grad else

        with open('test_samples\HR_PD_500_0_x2.pt', 'rb') as _f:
            pd_lr = np.expand_dims(pickle.load(_f), axis=(0, 1))

        pd_lr = np.float32(pd_lr)

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: pd_lr}
        ort_outs = ort_session.run(None, ort_inputs)

        print(ort_outs[0][0, 0].shape)

    def __process(self,  ):
        try:
            for i in range(30):
                if self.stop_thread is True:
                    print("Self stop_thread", self.stop_thread)
                    break
                # time.sleep(1)
                self.__test_onnx()
                self.counter += 1

            self.button_start_processing.config(state='normal')
            self.counter = 0
        except Exception as ex:
            print(ex)
            self.__shown_text = "An exception occurred!"


    def __start_processing(self):

        self.button_start_processing.config(state='disabled')

        self.processing_thread = threading.Thread(target=self.__process,
                                                  args=(), daemon=True)
        self.processing_thread.start()