import numpy as np
import matplotlib.pyplot as plt


def visualize_weight(weight, savename='weight'):
    ymax = (weight.shape[0] / 2, weight.shape[1] / 2)
    weights = weight.reshape(weight.shape[0] * weight.shape[1])
    wmax = np.fabs(weights).max()
    gapdif = weight.shape[0] / weight.shape[1]
    x = [0] * weight.shape[0]
    y = np.arange(ymax[0], ymax[0] - weight.shape[0], -1)
    plt.plot(x, y, 'o')
    x = [5] * weight.shape[1]
    y = np.arange(ymax[1] * gapdif, (ymax[1] -
                                     weight.shape[1]) * gapdif, -gapdif)
    plt.plot(x, y, 'o')
    for j in range(weight.shape[1]):
        for i in range(weight.shape[0]):
            color = 'r' if weight[i][j] > 0 else 'b'
            plt.plot([0, 5], [ymax[0] - i, (ymax[1] - j) * gapdif],
                     color=color, lw=np.fabs(weight[i][j]) / wmax)
        plt.savefig('{}{}.pdf'.format(savename, j))
        del plt.gca().lines[-weight.shape[0]:]
