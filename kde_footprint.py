import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde


class KdeRunner(object):
    def __init__(self, x_c, y_c, x_f, y_f, xi, yi):
        self.x_c = x_c
        self.y_c = y_c
        self.x_f = x_f
        self.y_f = y_f
        self.xi = xi
        self.yi = yi


    def perform_kde(self, x_, y_):


        ki = kde.gaussian_kde([x_, y_])
        zi = np.array(ki(np.vstack([self.xi.flatten(), self.yi.flatten()])))
        return zi

    def origin_x(self):
        #print(len(self.x_f))
        #zi_list = np.asarray([self.perform_kde(i, self.y_f[n][N]) for n, i in enumerate(self.x_f)  for N, I in enumerate(i) if len(I) > 2])
        xf_zero = [i for N, I in enumerate(self.x_f) for i in I[0]]
        yf_zero = [i for N, I in enumerate(self.y_f) for i in I[0]]
        zi_zero = self.perform_kde(xf_zero, yf_zero)


        first_frame = zi_zero.reshape(self.xi.shape)
        #plt.pcolormesh(self.xi, self.yi, first_frame, alpha=1, cmap=plt.cm.Spectral_r)
        #plt.show()

        origin = (np.where(first_frame > np.max(first_frame)*0.95))

        return np.min(self.xi[origin])

    def plotter(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c) if len(i) > 2])

        origin_x = self.origin_x()

        zee = None

        for i in zi_list:
            if zee is None:
                zee = i.copy()
            else:
                zee = zee + i

        plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=0.2, cmap=plt.cm.Spectral_r)

        # position of densities above 0.25std the mean
        # maxx = [self.xi[np.where(i.reshape(self.xi.shape) > (np.mean(i) + np.std(i) * 0.2))] for i in zi_list]
        # maxy = [self.yi[np.where(i.reshape(self.xi.shape) > (np.mean(i) + np.std(i) * 0.2))] for i in zi_list]
        maxx = [self.xi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.25)] for i in zi_list]
        maxy = [self.yi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.25)] for i in zi_list]

        maxx = np.array(maxx)
        maxy = np.array(maxy)

        for n, i in enumerate(maxx):
            ply = np.polyfit(i[0:10], maxy[n][0:10], 1)
            print(ply)

        #polylines = [np.poly1d(np.polyfit([i for i in maxx[N]], [i for i in maxy[N]], 1)) for N, I in enumerate(maxx)]

        lines = np.linspace(0, 1200, 1201)

        plt.scatter(maxx[0], maxy[0])
        plt.scatter(maxx[1], maxy[1])
        plt.show()
