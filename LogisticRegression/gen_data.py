import numpy as np
import csv


def GenerateData(centerX, centerY, r, label, ndata):
    X, Y = np.random.randn(ndata), np.random.rand(ndata)
    X = (X / np.fabs(X).max()) * r + centerX
    Y = (Y / np.fabs(Y).max()) * r + centerY
    return np.dstack((X, Y, np.array([label] * ndata)))

if __name__ == '__main__':
    data1 = GenerateData(1, 1, 2, 0, 100)
    data2 = GenerateData(2, 2, 2, 1, 100)
    data = np.hstack((data1, data2))[0]
    np.random.shuffle(data)
    data = [[d1, d2, int(l)] for d1, d2, l in data]

    with open('datas.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
