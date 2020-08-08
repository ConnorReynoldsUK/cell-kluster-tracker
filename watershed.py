import cv2
import numpy as np

class WatershedTransform(object):
    def __init__(self, img_frame):
        self.img_frame = img_frame

    def run_watershed(self):
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(self.img_frame, cv2.MORPH_OPEN, kernel, iterations=0)

        # sure background area
        sure_bg = cv2.dilate(self.img_frame, kernel, iterations=2)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, .1 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        self.img_frame = cv2.cvtColor(self.img_frame, cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(self.img_frame, markers)
        self.img_frame[markers == -1] = [0, 0, 0]
        thresh = cv2.cvtColor(self.img_frame, cv2.COLOR_RGB2GRAY)

        return thresh