import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class FullScreen:
    """Full-screen with Tkinter backend"""

    def __init__(self):
        self._root = tk.Tk()
        self._root.config(cursor="none")
        self._root.wm_attributes("-fullscreen", True)

        # set initial image
        img_gray = np.full((self.height, self.width), 127, dtype=np.uint8)
        tk_img_gray = self._cvt_ndarray_to_tkimage(img_gray)
        self._label = tk.Label(self._root, image=tk_img_gray)
        self._label.pack()

        # set key event
        def key_event(event):
            self.__del__()

        self._label.bind("<Key>", key_event)
        self._label.focus_set()

        self._root.update()

    def __del__(self):
        self._root.destroy()

    @property
    def width(self):
        return self._root.winfo_width()

    @property
    def height(self):
        return self._root.winfo_height()

    @property
    def shape(self):
        return self.height, self.width, 3

    def imshow(self, image: np.ndarray):
        tk_img = self._cvt_ndarray_to_tkimage(image)

        self._label.image = tk_img
        self._label.configure(image=tk_img)

        self._root.update_idletasks()
        self._root.update()

    def _cvt_ndarray_to_tkimage(self, image: np.ndarray) -> ImageTk.PhotoImage:
        """Convert ndarray data to PhotoImage using PIL"""
        if image.ndim == 2:
            img_rgb = np.dstack([image] * 3)  # Gray -> RGB
        else:
            img_rgb = image[:, :, ::-1]  # BGR -> RGB

        pil_img = Image.fromarray(img_rgb, mode="RGB")
        if pil_img.size != (self.width, self.height):
            pil_img = pil_img.resize((self.width, self.height), Image.NEAREST)

        tk_img = ImageTk.PhotoImage(pil_img)
        return tk_img
