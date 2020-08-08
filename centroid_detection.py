import cv2
import numpy as np

class CentroidDetector(object):
    def __init__(self, img_frame):
        self.img_frame = img_frame

    def moments(self, cent):
        M = cv2.moments(cent)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        return (cX, cY)

    def find_contours(self):
        frame = self.img_frame

        contours, hier = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        centroids = np.array([self.moments(i) for i in contours])
        return centroids