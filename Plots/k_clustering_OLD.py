import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import kde
import time
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

class KMeansClustering(object):
    def __init__(self, centroids_input, k_number):
        self.centroids_input = centroids_input
        self.k_number = k_number

    #def find_centres(self, x_val, y_val):

    def k_means_runner(self):

        centroids = self.centroids_input

        centroid_lens = np.asarray([len(i) for i in centroids])
        centroids_flat = np.asarray([c for C in centroids for c in C])
        centroids_flat_frame = np.asarray([n for n, i in enumerate(centroids) for N in i])

        centroids_x = np.asarray([[c[0] for c in C if (0, 0) not in c] for C in centroids])
        centroids_y = np.asarray([[c[1] for c in C if (0, 0) not in c] for C in centroids])



        colors = np.asarray(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        #print(centroids_x)


        km = KMeans(n_clusters=self.k_number)
        clustered = km.fit_predict(centroids_flat)
        clusters = [centroids_flat[np.where(clustered == i)] for i in range(max(clustered)+1)]

        centroids_col = [colors[I] for I in clustered]


        return centroids_x, centroids_y, clusters

    def k_means_plotter(self):
        tim = time.time()
        x_, y_, clusters = self.k_means_runner()


        x_flatt = np.asarray([i for I in x_ for i in I])
        y_flatt = np.asarray([i for I in y_ for i in I])

        x_cluster = np.asarray([[i[0] for i in I if (0, 0) not in i] for I in clusters])
        y_cluster = np.asarray([[i[1] for i in I if (0, 0) not in i] for I in clusters])

        xi, yi = np.mgrid[x_flatt.min():x_flatt.max():200 * 1j, y_flatt.min():y_flatt.max():200 * 1j]
        zi = None
        for n, i in enumerate(x_cluster):
            if len(i) != min([len(I) for I in x_cluster]):
                x_flat = np.asarray(i)
                y_flat = np.asarray(y_cluster[n])

                k = kde.gaussian_kde([x_flat, y_flat])
                if zi is None:
                    zi = np.array(k(np.vstack([xi.flatten(), yi.flatten()])))
                    print(len(zi))
                else:
                    #zi = zi+k(np.vstack([xi.flatten(), yi.flatten()]))+k(np.vstack([xi.flatten(), yi.flatten()]))
                    zi = zi+k(np.vstack([xi.flatten(), yi.flatten()]))

        x_two = np.asarray([i for i in x_cluster if len(i) != min([len(I) for I in x_cluster])])
        y_two = np.asarray([i for i in y_cluster if len(i) != min([len(I) for I in x_cluster])])

        models = [np.poly1d(np.polyfit(i, y_two[n], 3)) for n, i in enumerate(x_two)]
        #print([i for I in x_two for i in I])
        mid_line = np.poly1d(np.polyfit([i for I in x_two for i in I], [i for I in y_two for i in I], 1))
        lines = np.linspace(500, 1200, 5)
        line_all = np.linspace(min(x_flatt), max(x_flatt))

        #print(len(x_two), len(y_two))

        # for n, i in enumerate(x_cluster):
        #     x_flat = np.asarray(i)
        #     y_flat = np.asarray(y_cluster[n])
        #
        #     k = kde.gaussian_kde([x_flat, y_flat])
        #     #xi, yi = np.mgrid[x_flat.min():x_flat.max():x_flat.size*50*1j, y_flat.min():y_flat.max():y_flat.size*50*1j]
        #     xi, yi = np.mgrid[x_flatt.min():x_flatt.max():200*1j, y_flatt.min():y_flatt.max():200*1j]
        #
        #     zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        #     zip.append(zi)

        # xi, yi = np.mgrid[x_flat.min():x_flat.max():x_flat.size*50*1j, y_flat.min():y_flat.max():y_flat.size*50*1j]
        #xi, yi = np.mgrid[x_flat.min():x_flat.max():200 * 1j, y_flat.min():y_flat.max():200 * 1j]
        fig = plt.figure()
        fig.subplots_adjust(hspace=0, wspace=0)
        ax1 = plt.subplot2grid((5, 5), (0,0), colspan=4, rowspan=4)
        ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1, cmap=plt.cm.Spectral_r)
        ax1.plot(line_all, mid_line(line_all), linewidth=5, linestyle='--', color='k')

        for i in models:
            print(i(lines)[0])
            ax1.plot(lines, i(lines), linewidth=3, color='k')
            #adds gridd
            # #plt.arrow(max(lines), i(max(lines)), 0.4, 0.7, shape='full', head_width=1, length_includes_head=False, color='k')
            # plt.scatter(lines, i(lines), s=20, c='k', marker='o')
            # for n, I in enumerate(lines):
            #     if n < len(lines)-1:
            #         plt.plot([I, I], [i(I), mid_line(I)], linewidth=2, color='k', linestyle=':')
            #         #plt.plot([I, lines[n+1]], [mid_line(I), i(lines[n+1])], linewidth=4, color='k', linestyle=':')
            #         plt.plot([I, lines[n+1]], [i(I), mid_line(lines[n+1])], linewidth=4, color='k', linestyle=':')


        #plt.pcolormesh(xi.flatten(), yi.flatten(), zi, alpha=1)
        #find the density curve across x
        zee_x = None
        zee_y = None

        for i in zi.reshape(xi.shape).transpose():
            if zee_x is None:
                zee_x = i.copy()

            else:
                zee_x = zee_x + i.copy()

        for i in zi.reshape(yi.shape):

            if zee_y is None:
                zee_y = i.copy()
            else:
                zee_y = zee_y + i.copy()
        #find the median and iqr





        #plot a density curve
        ax2 = plt.subplot2grid((5, 5), (4, 0), rowspan=1, colspan=4, sharex=ax1)
        sum_x = [np.sum(i) for i in zi.reshape(xi.shape)]
        medx = (np.median(sum_x))
        q75x, q25x = np.percentile(sum_x, [75, 25])
        iqr = q75x - q25x


        sum_y = [np.sum(i) for i in zi.reshape(xi.shape).transpose()]
        medy = (np.median(sum_y))
        q75y, q25y = np.percentile(sum_y, [75, 25])
        iqr = q75x - q25x

        xi_plot = xi.transpose()[0]
        yi_plot = yi[0]

        print(sum_y)
        ax2.plot(xi.transpose()[0], sum_x)
        ax2.hlines(medx, xmin=0, xmax=np.max(xi), colors='k')
        ax2.hlines(q25x, xmin=0, xmax=np.max(xi), colors='k', linestyles=':')
        ax2.hlines(q75x, xmin=0, xmax=np.max(xi), colors='k', linestyles=':')
        #plt.plot(xi, zee)
        #fig = plt.figure()
        #plt.hexbin(x_flat, y_flat, gridsize=30)
        #ax.view_init(30, 80)

        ax3 = plt.subplot2grid((5,5), (0,4), rowspan=4, colspan=1, sharey=ax1)

        ax3.plot(sum_y, yi[0])
        ax3.vlines(medy, ymin=0, ymax=np.max(yi), colors='k')
        ax3.vlines(q25y, ymin=0, ymax=np.max(yi), colors='k', linestyles=':')
        ax3.vlines(q75y, ymin=0, ymax=np.max(yi), colors='k', linestyles=':')

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        yticks = ax3.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        plt.tight_layout()


        plt.show()


        xi_plot = xi.transpose()[0]
        yi_plot = yi[0]

        thresh_x = [xi_plot[n] for n, i in enumerate(sum_x) if i > medx]
        thresh_y = [yi_plot[n] for n, i in enumerate(sum_y) if i > medy]



        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_trisurf(xi.flatten(), yi.flatten(), zi, cmap=plt.cm.Spectral_r)
        # for an in range(135, 360):
        #
        #     ax.view_init(30, an)
        #     plt.show()