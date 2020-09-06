import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import math
from scipy.spatial import distance
import csv

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
        zee = None

        first_frame = zi_zero.reshape(self.xi.shape)
        plt.pcolormesh(self.xi, self.yi, first_frame, alpha=1, cmap=plt.cm.Spectral_r)
        plt.show()

        # mid_x = [np.mean(self.xi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.5)]) for i in zi_zero]
        # mid_y = [np.mean(self.yi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.5)]) for i in zi_zero]
        # mid = np.poly1d(np.polyfit([i for i in mid_x], [i for i in mid_y], 1))


        origin = (np.where(first_frame > np.max(first_frame)*0.9))

        return np.min(self.xi[origin])

    def mid_line(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c)])
        zee = None

        for i in zi_list:
            if zee is None:
                zee = i.copy()
            else:
                zee = zee + i

        x_pos = [self.xi[np.where(zee.reshape(self.xi.shape) > np.max(zee)*0.25)]]
        y_pos = [self.yi[np.where(zee.reshape(self.xi.shape) > np.max(zee)*0.25)]]
        print(x_pos)
        mid = np.poly1d(np.polyfit([i for i in x_pos[0]], [i for i in y_pos[0]], 1))
        return mid


    def plotter_pf(self):

        weight, mid = self.origin_x()

        #print(self.x_f, self.y_f)
        threshold = 0.05
        zi_list = [[self.perform_kde(self.x_f[n][N], self.y_f[n][N]) for N, I in enumerate(i) if len(I) > 2] for n, i in enumerate(self.x_f)]

        zi_maxx = [[self.xi[np.where(i.reshape(self.xi.shape) > np.max(I)*threshold)] for i in I] for I in zi_list]
        zi_maxy = [[self.yi[np.where(i.reshape(self.xi.shape) > np.max(I)*threshold)] for i in I] for I in zi_list]

        flat_maxx = [[i for I in II for i in I] for II in zi_maxx]
        flat_maxy = [[i for I in II for i in I] for II in zi_maxy]

        # mid = self.mid_line()
        # weight = self.origin_x()
        origin = mid(weight)

        flat_maxx = [[weight]*int(len(i)*5) + i for i in flat_maxx]
        flat_maxy = [[origin]*int(len(i)*5) + i for i in flat_maxy]

        #flat_maxx = [i for i in flat_maxx]
        #flat_maxy = [i for i in flat_maxy]

        line = [np.poly1d(np.polyfit([i for i in flat_maxx[n]], [i for i in flat_maxy[n]], 2)) for n, i in enumerate(flat_maxy)]

        print(weight, origin)
        lines = np.linspace(500, 1200)

        zee = None

        for n, i in enumerate(zi_list):
            for I in i:
                if zee is None:
                    zee = I.copy()
                else:
                   zee = zee + I

        plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)




        for n, i in enumerate(flat_maxx):
            #weighted = [[[weight[0]]]*len(i)]

            #weighted.append(lines)
            #print(weighted)
            plt.plot(lines, line[n](lines))
            plt.plot(lines, mid(lines))
            #lines = np.linspace(weight, 1200)
        plt.show()


        # for n, I in enumerate(zi_list):
        #     if zee is None:
        #         zee = I.copy()
        #     else:
        #         zee = zee + I
        #     plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)
        #     plt.savefig('C:/Users/cjrri/PycharmProjects/Trackz/Plots/Exp2_k_means/thresholding/%s.png' % format(
        #         np.around(n, 2), '.2f'))



    def plotter(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c) if len(i) > 2])

        origin_x = self.origin_x()

        maxx = [self.xi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.25)] for i in zi_list]
        maxy = [self.yi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.25)] for i in zi_list]

        polylines = [np.poly1d(np.polyfit([i for i in maxx[N]], [i for i in maxy[N]], 1)) for N, I in enumerate(maxx)]
        lines = np.linspace(origin_x, 1200, 1200)
        mid = np.poly1d(np.polyfit([i for i in lines], [(polylines[0](i)+polylines[1](i))/2 for i in lines], 1))
        origin_y = mid(origin_x)
        print(origin_y)
        # for i__ in np.linspace(0, 0.9, 10):
        #     for i_ in np.linspace(0, 0.9, 10):
        #         if i_ >= i__:
        thresh = np.linspace(0.3, 0.9, 80)
        thresh_x = [[self.xi[np.where(i.reshape(self.xi.shape) > np.max(i)*t)] for t in thresh] for i in zi_list]
        thresh_y = [[self.yi[np.where(i.reshape(self.xi.shape) > np.max(i)*t)] for t in thresh] for i in zi_list]

        threshxx = [[i for I in thresh_x[N] for i in I] for N, II in enumerate(thresh_x)]
        threshyy = [[i for I in thresh_y[N] for i in I] for N, II in enumerate(thresh_y)]

        thr_c = threshxx.copy()

        #filter out x before origin
        for n, e in enumerate(threshxx):
            for N, E in enumerate(e):
                if E <= origin_x:
                    thr_c[n][N] = origin_x.copy()

        for polyy in np.linspace(1, 10, 10):
            #anchor the origin into the threshold position
            threshx_ori = [([origin_x]*len(i)*5) + i for i in thr_c]
            threshy_ori = [([origin_y]*len(i)*5) + i for i in threshyy]
            polylines2 = [np.poly1d(np.polyfit([i for i in threshx_ori[N]], [i for i in threshy_ori[N]], polyy)) for N, I in enumerate(thr_c)]
            #polylines2 = [np.poly1d(np.polyfit([i for i in threshxx[N]], [i for i in threshyy[N]], 2)) for N, I in enumerate(threshxx)]

            angle_pointsx = np.linspace(origin_x, 1000)
            angle_pointsy = mid(angle_pointsx)

            polyfill = [[[int(i), int(I(i))] for i in lines] for I in polylines2]
            points0 = []
            points0_num = []

            points1 = []
            points1_num = []

            #####ANGLES
            mid_zero = [0, mid(0)]
            mid_zero_plus = [100, mid(100)]
            mid_zero_straight = [100, mid(0)]
            opp = mid(100)-mid(0)

            mid_angle = (math.degrees((math.atan(opp/100))))



            for n, i in enumerate(angle_pointsx):
                for I in range(0, 1000):
                    endy = angle_pointsy[n] + I * math.sin(math.radians(90))
                    endx = angle_pointsx[n] + I * math.cos(math.radians(90))


                    if [int(endx), int(endy)] in polyfill[0]:
                        print(endx, endy)
                        points0.append([int(endx), int(endy)])
                        points0_num.append([angle_pointsx[n], angle_pointsy[n]])
                    if [int(endx), int(endy)] in polyfill[1]:
                        points1.append([int(endx), int(endy)])
                        points1_num.append([angle_pointsx[n], angle_pointsy[n]])









            zee = None

            for i in zi_list:
                if zee is None:
                    zee = i.copy()
                else:
                    zee = zee + i

            # midx = self.xi[np.where(zee.reshape(self.xi.shape) > np.max(zee)*0.00001)]
            # midy = self.yi[np.where(zee.reshape(self.xi.shape) > np.max(zee)*0.00001)]
            #
            #
            #
            # line = np.poly1d(np.polyfit([i for i in midx], [i for i in midy], 1))
            print((polylines2[0]-polylines2[1].roots))

            plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)
            for i in polylines2:
                plt.plot(lines, i(lines))
            plt.plot(lines, mid(lines))
            final_angles = []
            final_anlges_x = []
            for n, i in enumerate(points0):
                if n >= 1:
                    ori = (origin_x, origin_y)
                    opp_p = (points0[n][0], points0[n][1])
                    adj_p = (points0_num[n][0], points0_num[n][1])



                    adj_d = distance.euclidean(ori, adj_p)
                    opp_d = distance.euclidean(adj_p, opp_p)
                    hyp_d = distance.euclidean(ori, opp_p)
                    #print(11111, adj_d)
                    #print(opp_d, hyp_d, opp_d/hyp_d, 1111)


                    #fin_angle = math.degrees(math.atan2(opp_p[1] - ori[1], opp_p[0] - ori[0])) - math.degrees(math.atan2(adj_p[1] - ori[1], adj_p[0] - ori[0]))
                    #fin_angle = math.degrees(math.acos(((opp_d**2) - (adj_d**2) - (hyp_d**2))/(-2*adj_d*hyp_d)))
                    if 0 not in (adj_d, opp_d, hyp_d):
                        fin_angle = math.degrees(math.acos(((hyp_d**2) + (adj_d**2) - (opp_d**2))/(2*adj_d*hyp_d)))

                        final_angles.append(fin_angle)
                        final_anlges_x.append(i[0])
                        print(fin_angle, i[0])
                    #PLOT THE ANGLE LINES
                    #plt.plot([points0_num[n][0], points0[n][0]], [points0_num[n][1], points0[n][1]])



            plt.xlim(0, np.max(self.xi))
            plt.ylim((0, np.max(self.yi)))
            # plt.title('Min: %s    Max: %s' % (format(np.around(i__, 2), '.2f'), format(np.around(i_, 2), '.2f')))
            plt.xlabel(xlabel='X-position')
            plt.ylabel(ylabel='Y-position')
            plt.show()
            # plt.savefig('C:/Users/cjrri/PycharmProjects/Trackz/Plots/Exp3/%s_%s.png' % (format(np.around(i__, 2), '.2f'), format(np.around(i_, 2), '.2f')))
            #plt.clf()

            with open('exp13.csv', 'w', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(final_angles)
                wr.writerow(final_anlges_x)

            #
            #
            # plt.plot(final_anlges_x, final_angles)
            # plt.xlabel(xlabel='X-position')
            # plt.ylabel(ylabel='Angle')
            #
            # plt.show()

    def new_plotter(self):
        zi_list = np.asarray([self.perform_kde(i, self.y_c[n]) for n, i in enumerate(self.x_c) if len(i) > 2])

        origin_x = self.origin_x()

        zee = None

        for i in zi_list:
            if zee is None:
                zee = i.copy()
            else:
                zee = zee + i
        plt.pcolormesh(self.xi, self.yi, zee.reshape(self.xi.shape), alpha=1, cmap=plt.cm.Spectral_r)

        maxx = [self.xi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.25)] for i in zi_list]
        maxy = [self.yi[np.where(i.reshape(self.xi.shape) > np.max(i)*0.25)] for i in zi_list]

        polylines = [np.poly1d(np.polyfit([i for i in maxx[N]], [i for i in maxy[N]], 1)) for N, I in enumerate(maxx)]
        lines = np.linspace(origin_x, 1200, 1200)
        mid = np.poly1d(np.polyfit([i for i in lines], [(polylines[0](i)+polylines[1](i))/2 for i in lines], 1))
        origin_y = mid(origin_x)
        print(origin_x, origin_y)
        #set a minimum and maximum threshold
        thresh = np.linspace(0.5, 0.99, 80)

        thresh_x = [[self.xi[np.where(i.reshape(self.xi.shape) > np.max(i) * t)] for t in thresh] for i in zi_list]
        thresh_y = [[self.yi[np.where(i.reshape(self.xi.shape) > np.max(i) * t)] for t in thresh] for i in zi_list]

        # thresh_x = [[self.xi[np.where(i.reshape(self.xi.shape) > np.max(i) * t)] for t in thresh] for i in zi_list]
        # thresh_y = [[self.yi[np.where(i.reshape(self.xi.shape) > np.max(i) * t)] for t in thresh] for i in zi_list]
        #
        threshxx = [[int(i) for I in thresh_x[N] for i in I] for N, II in enumerate(thresh_x)]
        threshyy = [[int(i) for I in thresh_y[N] for i in I] for N, II in enumerate(thresh_y)]

        mean_x = [[I for I in np.unique(threshxx[n])] for n, II in enumerate(threshxx)]
        mean_y = [[np.mean([threshyy[n][i] for i in np.where(threshxx[n] == I)[0]]) for I in np.unique(threshxx[n])] for n, II in enumerate(threshxx)]
        #print(len(mean_y[0]), len(mean_x[0]))

        # threshyy = [[int(i) for I in thresh_y[N] for i in I] for N, II in enumerate(thresh_y)]
        #
        threshx_ori = [([int(origin_x)] * len(mean_x)) + i for i in mean_x]
        threshy_ori = [([int(origin_y)] * len(mean_x) ) + i for i in mean_y]
        #
        # threshx_ori = threshxx
        # threshy_ori = threshyy
        #
        zipped_points = [list(zip(threshx_ori[n], threshy_ori[n])) for n, i in enumerate(threshx_ori)]
        print(zipped_points)
        import test
        # from scipy.special import comb
        for i in zipped_points:

            berzie = (test.bezier_curve(i, 500))
            plt.plot(berzie[0], berzie[1], c='k')


        plt.show()
