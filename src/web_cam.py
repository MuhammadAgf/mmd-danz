from urllib.request import urlopen
import cv2
import numpy as np


class WebCam:
    def __init__(self, host):
        self.host = host

    def get(self, readFlag=cv2.IMREAD_COLOR):
        resp = urlopen(self.host)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        return image
