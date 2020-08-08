import cv2
import numpy as np
import time
import movie_reader, laplacian_fitler, watershed, centroid_detection, cluster_centroids, run_kde
import os

test = movie_reader.MoviePlayer("%s/Data/Movies/Exp2_GFP-271017-11_2.avi" % os.getcwd())

def run_detection(img_frame):

    new_frame = laplacian_fitler.LaplacianSegmenter(img_frame, 60, 0).apply_filter()
    new_frame = watershed.WatershedTransform(new_frame).run_watershed()
    centroids = centroid_detection.CentroidDetector(new_frame).find_contours()

    return centroids

frames, frame_width, frame_height = test.read_movie()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter("a.avi", fourcc, 20.0, (frame_width, frame_height))
frames_filtered = frames.copy()

prev_time = time.time()
cents = [run_detection(i) for i in frames]
print(time.time()-prev_time)

#k_clustering.KMeansClustering(cents, 3).k_means_plotter()
#kde_plotter.KdePlotter(cents).k_cluster()
x_, y_, xi, yi = cluster_centroids.PerformCluster(cents).export_clusters()
run_kde.KdeRunner(x_, y_, xi, yi).plotter()
# for n, frame in enumerate(frames):
#     new_frame = laplacian_fitler.LaplacianSegmenter(frame, 60, 0).apply_filter()
#     new_frame = watershed.WatershedTransform(new_frame).run_watershed()
#     frames_filtered[n] = new_frame
#     print(n)
#     print(centroid_detection.CentroidDetector(new_frame).find_contours())



    #cv2.imshow('fr', new_frame)
    #cv2.waitKey(1)

