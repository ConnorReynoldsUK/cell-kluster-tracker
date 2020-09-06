import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class PerformCluster(object):
    def __init__(self, centroids_input):
        self.centroids_input = centroids_input
        #self.centroids_input = self.centroids_input[0:int(len(self.centroids_input)*0.7)]
        self.k_number = None

    def km_cluster(self, centroids_flat):

        km = KMeans(n_clusters=3)
        clustered = km.fit_predict(centroids_flat)
        return clustered

    def max_intracluster_std(self, x_):
        std_total = [np.sum([np.std(i + x_[a]) for a in range(0, self.k_number) if a != n]) for n, i in enumerate(x_)]
        return np.where(std_total == np.max(std_total))[0]

    def cent_perframe(self, centroids, clusters, max_std):
        frame_list = np.array([n for n, C in enumerate(centroids) for c in C])

        frame_clusters = [clusters[np.where(frame_list == i)] for i in range(0, np.max(frame_list))]
        centroid_pcluster_pframe = [[[centroids[NN][N] for N, i in enumerate(II) if i == n and (0, 0) not in centroids[NN][N]] for n in
             range(0, np.max(clusters)+1)] for NN, II in enumerate(frame_clusters)]

        centroid_pframe_pcluster = [[i[n] for i in centroid_pcluster_pframe] for n in range(0, self.k_number) if n != max_std]

        x_perframe_percluster = [[[II[0] for II in I] for I in i] for i in centroid_pframe_pcluster]
        y_perframe_percluster = [[[II[1] for II in I] for I in i] for i in centroid_pframe_pcluster]
        return x_perframe_percluster, y_perframe_percluster

    def export_clusters(self):
        centroids = [[i for i in I if (0, 0) not in i] for I in self.centroids_input]
        centroids_flat = [i for I in centroids for i in I]

        clusters = self.km_cluster(centroids_flat)
        self.k_number = np.max(clusters) +1
        centroids_clustered = [[centroids_flat[n] for n, i in enumerate(clusters) if i == I] for I in
                                    range(0, self.k_number)]
        max_x = max([i[0] for i in centroids_flat])
        max_y = max([i[1] for i in centroids_flat])
        xi, yi = np.mgrid[0:max_x:250 * 1j, 0:max_y:250 * 1j]


        max_std_ind = self.max_intracluster_std([[i[0] for i in I] for I in centroids_clustered])
        x_percluster = [[i[0] for i in I] for n, I in enumerate(centroids_clustered) if n != max_std_ind]
        y_percluster = [[i[1] for i in I] for n, I in enumerate(centroids_clustered) if n != max_std_ind]

        x_perframe, y_perframe = self.cent_perframe(centroids, clusters, max_std_ind)
        print(len(y_perframe), len(y_perframe[0]))

        for n, i in enumerate(x_percluster):


            plt.scatter(x_percluster[n], y_percluster[n], alpha=1, linewidths=0.1)
        plt.xlim(0, 1600)
        plt.ylim(0, 1200)
        plt.show()
        plt.clf()

        return x_percluster, y_percluster, x_perframe, y_perframe, xi, yi


        #x, y = self.cent_perframe()
        #print(len(x))
