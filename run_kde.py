import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

class KdeRunner(object):
    def __init__(self, x_c, y_c, xi, yi):
        self.x_c = x_c
        self.y_c = y_c
        self.xi = xi
        self.yi = yi


    def perform_kde(self, x_, y_):
        ki = kde.gaussian_kde([x_, y_])
        zi = np.array(ki(np.vstack([self.xi.flatten(), self.yi.flatten()])))
        return zi

    def plotter(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c)])

        zee = None

        for i in zi_list:
            if zee is None:
                zee = i.copy()
            else:
                zee = zee + i
        plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)
        plt.show()


