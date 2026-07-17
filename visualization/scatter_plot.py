import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

import utils.constants as cs


def plot(title, X, labels=None, plot=True, marker='o', alpha=1):
    
    if plot:
        nrDim = len(X[0])
        fig = plt.figure() 
        plt.title(title)
        if nrDim == 2:
            if labels is None:
                plt.scatter(X[:, 0], X[:, 1], marker=marker, edgecolors='k')
            else:
                try:
                    label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
                except KeyError:
                    print('Too many labels! Using default colors...\n')
                    label_color = [l for l in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k', alpha=alpha)
        if nrDim == 3:
            ax = fig.add_subplot(projection='3d')
            ax.set_title(title)
            if labels is None:
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, edgecolors='k')
            else:
                try:
                    label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
                except KeyError:
                    print('Too many labels! Using default colors...\n')
                    label_color = [l for l in labels]
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label_color, marker=marker, edgecolors='k', alpha=alpha)


def plot_grid(title, X, pn, labels=None, plot=True, marker='o'):
    
    X = preprocessing.MinMaxScaler((0, 1)).fit_transform(X)
    if plot:
        nrDim = len(X[0])
        label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        plt.title(title)
        if nrDim == 2:
            ax = fig.gca()
            if not isinstance(pn, int):
                ax.set_xticks(np.arange(0, pn[0], 1))
                ax.set_yticks(np.arange(0, pn[1], 1))
            else:
                ax.set_xticks(np.arange(0, pn, 1))
                ax.set_yticks(np.arange(0, pn, 1))
            plt.scatter(X[:, 0], X[:, 1], marker=marker, c=label_color, s=25, edgecolor='k')
            plt.grid(True)
        if nrDim == 3:
            ax = Axes3D(fig)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.set_xticks(np.arange(0, pn, 1))
            # ax.set_zticks(np.arange(0, pn, 1))
            # ax.set_yticks(np.arange(0, pn, 1))
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, s=25)
            # plt.grid(True)
        # fig.savefig("cevajeg.svg", format='svg', dpi=1200)

