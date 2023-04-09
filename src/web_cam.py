import queue
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class WebCam:
    def __init__(self, host, is_stream=False):
        self.host = host
        self.is_stream = is_stream
        if self.is_stream:
            self.cap = VideoCapture(self.host)

    def get_stream(self):
        frame = self.cap.read()
        return frame

    def get_image(self):
        readFlag = cv2.IMREAD_COLOR
        resp = urlopen(self.host)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        return image

    def get(self):
        if self.is_stream:
            return self.get_stream()
        else:
            return self.get_image()
