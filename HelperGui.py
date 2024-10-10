import customtkinter as ctk
import tkinter

from customtkinter import CTkFrame


class HelperGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1250x800")
        self.title("NeuralNetwork")

        self.frame_left = ctk.CTkFrame(self, height = int(self._max_height/2), width = int(self._max_width/2))
        self.frame_left.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = "nsew")

        input = txt_slider(self.frame_left,"Input Neurons", (0, 0), 1, 20)
        output = txt_slider(self.frame_left,"Output Neurons", (1, 0), 1, 20)
        hidden = txt_slider(self.frame_left,"Hidden Layers", (2, 0), 1, 20)
        hidden_layer = hidden_layers_input(self.frame_left, (3, 0))


class txt_slider(ctk.CTkFrame):
    def __init__(self, frame, text, rc, min, max):
        super().__init__(frame)
        self.grid(row = rc[0], column = rc[1], padx = 10, pady = 10, sticky = "ew")
        self.label = ctk.CTkLabel(self, text = text)
        self.label.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = "w")
        self.slider = ctk.CTkSlider(self, width = 500, from_ = min, to = max,number_of_steps = max-min, command = self.update_text)
        self.slider.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = "ew")
        self.slider.set(1)
        self.label2 = ctk.CTkLabel(self, text = "0")
        self.label2.grid(row = 0, column = 2, padx = 10, pady = 10, sticky = "e")
        self.label2.configure(text = int(self.slider.get()))

    def update_text(self, value):
        self.label2.configure(text = int(self.slider.get()))

class hidden_layers_input(ctk.CTkFrame):
    def __init__(self, frame, rc):
        super().__init__(frame)
        self.grid_rowconfigure(0, weight = 1)
        self.grid(row = rc[0], column = rc[1], padx = 10, pady = 10, sticky = "ew")
        self.hidden_count = txt_slider(self,"Hidden Layers", (2, 0), 1, 20)
        self.hidden_count.grid(row = 0, column = 0, padx = 10, pady = 10)
        self.generate = ctk.CTkButton(self, text = "Generate", command = self.init_hidden_layers)
        self.generate.grid(row = 0, column = 1, padx = (0, 25), pady = 10, sticky = "ew")
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "nsew", columnspan = 2)
        self.hidden_layers = []

    def init_hidden_layers(self):
        self.hidden_count.label2.configure(text = int(self.hidden_count.slider.get()))
        for i in self.hidden_layers:
            i.destroy()
        for i in range(int(self.hidden_count.slider.get())):
            x = txt_slider(self.scroll_frame, f"Hidden Layer {i+1}", (i, 1), 1, 20)
            self.hidden_layers.append(x)



HelperGui().mainloop()