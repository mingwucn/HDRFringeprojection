import os
import sys
import tempfile
from multiprocessing import Process
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


def resize(image, size):
    import importlib

    is_pillow_available = importlib.util.find_spec("PIL") is not None
    width, height = size
    if is_pillow_available:
        from PIL import Image

        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((width, height), Image.NEAREST)
        return np.array(pil_img)
    else:
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


class FullScreen:
    """Full-screen with PyQt5 backend

    Caution! This FullScreen class is really TRICKY.
    """

    def __init__(self):
        app = QtWidgets.QApplication([])
        screen = app.primaryScreen()
        size = screen.size()
        self.width = size.width()
        self.height = size.height()

        self.tmpdir_img = tempfile.TemporaryDirectory()
        self.filename_img = os.path.join(self.tmpdir_img.name, "tmp_image.dat")
        self.img = np.memmap(
            self.filename_img, dtype=np.uint8, mode="w+", shape=self.shape
        )

        self.tmpdir_flag = tempfile.TemporaryDirectory()
        self.filename_flag = os.path.join(self.tmpdir_flag.name, "tmp_flag.dat")
        self.flag = np.memmap(
            self.filename_flag, dtype=np.uint8, mode="w+", shape=(1)
        )

        # set initial image
        img_gray = np.full(self.shape, 127, dtype=np.uint8)
        self.imshow(img_gray)

        # launch fullscreen app
        self.p = Process(target=self._launch_fullscreen_app)
        self.p.start()

    @property
    def shape(self):
        return self.height, self.width, 3

    def imshow(self, image):
        if image.ndim == 2:
            img_rgb = np.dstack([image] * 3)  # Gray -> RGB
        else:
            img_rgb = image[:, :, ::-1]  # BGR -> RGB

        if img_rgb.shape != self.shape:
            img_rgb = resize(img_rgb, (self.width, self.height))

        self.img[:] = img_rgb[:]
        self.flag[:] = True

    def _launch_fullscreen_app(self):
        class QWidgetFullScreen(QtWidgets.QLabel):
            def __init__(self, filename_img, shape, filename_flag):
                super().__init__()

                self.img = np.memmap(
                    filename_img, dtype=np.uint8, mode="r", shape=shape
                )

                self.height, self.width, ch = self.img.shape

                self.flag = np.memmap(
                    filename_flag, dtype=np.uint8, mode="r+", shape=(1)
                )

                self.update_image()

                f = 60.0  # update rate Hz
                timer = QtCore.QTimer(self)
                timer.timeout.connect(self.update_image)
                timer.start(1000.0 / f)  # ms

                self.setCursor(QtCore.Qt.BlankCursor)
                self.showFullScreen()

            def __del__(self):
                # Close mmap objects
                if hasattr(self, "img"):
                    self.img._mmap.close()
                    del self.img

                if hasattr(self, "flag"):
                    self.flag._mmap.close()
                    del self.flag

            def update_image(self):
                if self.flag:
                    qt_img = QtGui.QImage(
                        self.img.flatten(),
                        self.width,
                        self.height,
                        QtGui.QImage.Format_RGB888,
                    )

                    self.setPixmap(QtGui.QPixmap.fromImage(qt_img))

                    self.update()

                    self.flag[:] = False

        app = QtWidgets.QApplication([])
        fullscreen_widget = QWidgetFullScreen(
            self.filename_img, self.shape, self.filename_flag
        )
        sys.exit(app.exec_())

    def destroyWindow(self):
        self.__del__()

    def __del__(self):
        # Terminate fullscreen app
        if hasattr(self, "p"):
            if self.p is not None:
                self.p.terminate()
                del self.p

        # Close mmap objects
        if hasattr(self, "img"):
            self.img._mmap.close()
            del self.img

        if hasattr(self, "flag"):
            self.flag._mmap.close()
            del self.flag

        # Remove tmpfile
        if hasattr(self, "tmpdir_img"):
            self.tmpdir_img.cleanup()

        if hasattr(self, "tmpdir_flag"):
            self.tmpdir_flag.cleanup()
