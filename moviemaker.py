import cv2
import re
import numpy as np
import os

video = None

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

for root, dirs, files in os.walk('%s/Plots/Exp2_k_means/thresholding/frame_by_frame' % os.getcwd(), topdown=False):
    print(files)
    max_file = np.max([len(i) for i in files])
    files.sort(key=natural_keys)

    for i in files:

        print(root)
        img = cv2.imread(root+'/'+i)
        print(img.shape)
        if video is None:
            height, width, a = img.shape
            video = cv2.VideoWriter('plot_movie.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))

        video.write(img)

        #cv2.destroyWindow()
    video.release()




