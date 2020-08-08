import cv2
import numpy as np

from skimage import exposure

class MoviePlayer(object):

    def __init__(self, moviefile):
        self.moviefile = moviefile
        self.frames = None

    def read_movie(self):

        cap = cv2.VideoCapture(self.moviefile)
        movie_length = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        self.moviefile = np.array([np.zeros((frame_height, frame_width))]*max(movie_length), dtype=np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #out = cv2.VideoWriter("%s.avi" % self.title, fourcc, 20.0, (frame_width, frame_height))
        frame_no = 0

        success = cap.grab()
        fno = 0

        for fno in range(0, max(movie_length), 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.moviefile[fno] = frame

        return self.moviefile, frame_width, frame_height

test = MoviePlayer("Exp8_HH4_-0002_9.avi")
#test.read_movie()