import cv2
import numpy as np

class LaplacianSegmenter(object):

    def __init__(self, img_frame, central_value, blur=0):
        self.img_frame = img_frame
        self.central_value = central_value
        self.blur = blur


    def make_filter(self):

        k1, k2, v_min, v_max = 9, 9, -1, self.central_value
        conv_filter = np.array([[v_min] * k1] * k2)
        (conv_filter[int((k2 / 2))][int((k1 / 2))]) = v_max

        return conv_filter

    def apply_filter(self):
        frame = self.img_frame
        conv_filter = self.make_filter()

        if self.blur == 1:
            frame = cv2.medianBlur(frame, 3)

        frame = cv2.filter2D(frame, -1, conv_filter)
        _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return frame


test = LaplacianSegmenter(65, 1)
#test.make_filter()