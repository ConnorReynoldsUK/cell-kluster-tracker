import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import math
from scipy.spatial import distance
import csv
import statsmodels.api as sm

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
        plt.pcolormesh(self.xi, self.yi, first_frame, alpha=1, cmap=plt.cm.Spectral_r)
        plt.show()

        origin = (np.where(first_frame > np.max(first_frame)*0.95))

        return np.min(self.xi[origin])




    def new__plotter(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c) if len(i) > 2])

        origin_x = self.origin_x()

        zee = None

        for i in zi_list:
            if zee is None:
                zee = i.copy()
            else:
                zee = zee + i

        maxx = [self.xi[np.where(i.reshape(self.xi.shape) > (np.mean(i)+np.std(i)*0.1))] for i in zi_list]
        maxy = [self.yi[np.where(i.reshape(self.xi.shape) > (np.mean(i)+np.std(i)*0.1))] for i in zi_list]

        polylines = [np.poly1d(np.polyfit([i for i in maxx[N]], [i for i in maxy[N]], 1)) for N, I in enumerate(maxx)]
        lines = np.linspace(origin_x, 1200, 1200)
        mid = np.poly1d(np.polyfit([i for i in lines], [(polylines[0](i)+polylines[1](i))/2 for i in lines], 1))
        origin_y = mid(origin_x)
        print(origin_x, origin_y)


        #set a minimum and maximum threshold
        for i in np.linspace(0.0, 1.0, 20):
            thresh = np.linspace(i, 1, 80)

            print(zi_list[0])

            thresh_x = [[self.xi[np.where(i.reshape(self.xi.shape) > np.max(i) * t)] for t in thresh] for i in zi_list]
            thresh_y = [[self.yi[np.where(i.reshape(self.xi.shape) > np.max(i) * t)] for t in thresh] for i in zi_list]


            threshxx = [[int(i) for I in thresh_x[N] for i in I] for N, II in enumerate(thresh_x)]
            threshyy = [[int(i) for I in thresh_y[N] for i in I] for N, II in enumerate(thresh_y)]

            mean_x = [[I for I in np.unique(threshxx[n])] for n, II in enumerate(threshxx)]
            mean_y = [[np.mean([threshyy[n][i] for i in np.where(threshxx[n] == I)[0]]) for I in np.unique(threshxx[n])] for n, II in enumerate(threshxx)]

            #print(len(mean_y[0]), len(mean_x[0]))

            # threshyy = [[int(i) for I in thresh_y[N] for i in I] for N, II in enumerate(thresh_y)]
            #
            threshx_ori = [([int(origin_x)] * len(mean_x)) + i for i in mean_x]
            threshy_ori = [([int(origin_y)] * len(mean_x) ) + i for i in mean_y]

            zipped_points = [list(zip(threshx_ori[n], threshy_ori[n])) for n, i in enumerate(threshx_ori)]
            import test
            # from scipy.special import comb
            #plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)

            for I in zipped_points:

                berzie = (test.bezier_curve(I, 500))
                plt.plot(berzie[0], berzie[1], c='k')

            plt.plot(lines, mid(lines), c='k', linestyle=':')
            plt.title('Min threshold = %s' % (format(np.around(i, 2), '.2f')))
            #plt.savefig('C:/Users/cjrri/PycharmProjects/Trackz/Plots/Exp1/Thresholds/%s.png' % (format(np.around(i, 2), '.2f')))

            plt.show()

    def fit_polynomial(self, zip_list, order):
        poly_range = range(1, 10)
        print(zip_list)
        polylines2 = [[np.poly1d(np.polyfit([i[0] for i in I], [i[1] for i in I], num)) for I in zip_list] for num in poly_range]
        return polylines2

    def mid_line(self, x, y):
        poly = np.poly1d(np.polyfit(x, y, 1))
        return poly

    def density_plot(self, zi_max):
        return zi_max



    def new_plotter(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c) if len(i) > 2])

        origin_x = self.origin_x()

        zee = None

        for i in zi_list:
            if zee is None:
                zee = i.copy()
            else:
                zee = zee + i
        abundance = [np.sum(i) for i in zee.reshape(self.xi.shape)]
        ab_mean = np.mean(abundance)
        ab_std = np.std(abundance)
        print(ab_mean, ab_std)


        # plt.plot(self.xi, abundance)
        # plt.hlines(ab_mean, 0, np.max(self.xi))
        #
        # plt.hlines(ab_mean-(ab_std*0.25), 0, np.max(self.xi), colors='m')
        # plt.hlines(ab_mean + (ab_std * 0.25), 0, np.max(self.xi), colors='m')
        # plt.show()

        #position of densities above 0.25std the mean
        maxx = [self.xi[np.where(i.reshape(self.xi.shape) > (np.mean(i)+np.std(i)*0.1))] for i in zi_list]
        maxy = [self.yi[np.where(i.reshape(self.xi.shape) > (np.mean(i)+np.std(i)*0.1))] for i in zi_list]
        #print(maxx[0], maxx[1])
        #polylines = [self.mid_line(maxx[n], maxy[n]) for n, i in enumerate(maxx)]

        #create lines for the higher density positions then find the midpoint to determine the primitve streak
        #polylines = [np.poly1d(np.polyfit([ea for ea in maxx[1]], [ea for ea in maxy[1]], 1)) for N, I in enumerate(maxx)]
        polylines = np.poly1d(np.polyfit([i for i in maxx[0]], [i for i in maxy[0]], 1))

        lines = np.linspace(origin_x, np.mean(self.xi)+np.std(self.xi), 1200)
        print(polylines(lines))
        mid = np.poly1d(np.polyfit([i for i in lines], [(polylines[0](i)+polylines[1](i))/2 for i in lines], 1))
        #establish an origin using the midline
        origin_y = mid(origin_x)

        minimum_c = [np.min(i) for i in self.x_c]
        maximum_c = np.max([i for I in self.x_c for i in I])
        min_pos = [np.max((np.where(self.xi < minimum_c[n]))[0]) for n, i in enumerate(minimum_c)]

        intz = 10
        xi_range = [i for i in range(0, len(self.xi))]
        zi_max = [[np.mean(i) for i in I.reshape(self.xi.shape)] for I in zi_list]

        #zi_max = [[np.average(i, weights=i) for i in I.reshape(self.xi.shape)] for I in zi_list]
        colors = ['r', 'b']
        for n, i in enumerate(zi_max):
            mea = np.mean(i)
            print(mea, np.std(i))
            plt.plot(self.xi , i, c=colors[n])
            plt.hlines(mea, xmin=0, xmax=1400, colors=colors[n], linestyles=':')
            plt.title('Weighted average')
            #plt.yscale('log')
            plt.ylabel('Density')
            plt.xlabel('X position')
        plt.show()
        plt.clf()



        #print(self.xi[np.where(zi_list[0].reshape(self.xi.shape) > np.mean(zi_list[0]))])

        zi_mea = [[int(np.average(xi_range,  weights=I.reshape(self.xi.shape)[n])) for n in xi_range] for I in zi_list**2]
        #zi_mea = [[int(np.mean(xi_range[n])) for n in xi_range] for I in zi_list]

        weighted_x = [np.mean(i) for i in self.xi]
        weighted_y = [[(self.yi[0][i]) for i in I] for I in zi_mea]


        plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)
        plt.plot(lines, mid(lines), c='k', linestyle=':', linewidth=2)
        plt.scatter(origin_x, origin_y, c='y', linewidths=5, alpha=1)

        zipped_points = [list(zip(weighted_x[min_pos[n]:len(weighted_x)], weighted_y[n][min_pos[n]:len(weighted_x)])) for n, i in enumerate(weighted_y)]

        zipped_points = [[(origin_x, origin_y)]*len(i) + i for i in zipped_points]
        import test
        # from scipy.special import comb
        # for i in zipped_points:
        #     berzie = (test.bezier_curve(i, 50))
        #     plt.plot(berzie[0], berzie[1], c='k')

        for i in self.fit_polynomial(zipped_points, 3):
            for I in i:
                plt.plot(lines, I(lines))


        plt.plot(lines, mid(lines), c='k', linestyle=':')
        plt.xlabel('X position')
        plt.ylabel('Y position')


        plt.show()

