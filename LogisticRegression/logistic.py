import argparse
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


def plot_data(X, t):
    positive = [i for i in range(len(t)) if t[i] == 1]
    negative = [i for i in range(len(t)) if t[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1],
                c='red', marker='o', label='label 0')
    plt.scatter(X[negative, 0], X[negative, 1],
                c='blue', marker='o', label='label 1')
    plt.legend()

class PlotEveryEpoch(keras.callbacks.Callback):

    def __init__(self, X):
        self.X = X
        self.line = None

    def on_epoch_end(self, epoch, logs):
        if epoch < 100 or epoch % 10 == 0:
            weights = self.model.layers[0].get_weights()
            plt.figure(1)
            w1 = weights[0][0, 0]
            w2 = weights[0][1, 0]
            b = weights[1][0]
            # plot boundary
            xmin, xmax = min(self.X[:, 0]), max(self.X[:, 0])
            ymin, ymax = min(self.X[:, 1]), max(self.X[:, 1])
            xs = np.linspace(xmin, xmax, 100)
            ys = [- (w1 / w2) * x - (b / w2) for x in xs]
            if self.line is not None:
                self.line.remove()
            self.line, = plt.plot(xs, ys, 'b-', label='decision boundary')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.legend()
            plt.savefig("graph/learned_boundary_{0:03d}.png".format(epoch))


def main():
    # load training data
    data = np.genfromtxt('datas.csv', delimiter=',')
    X = data[:, (0, 1)]  # exstract 0th, 1st elements
    t = data[:, 2]
    X = preprocessing.scale(X)

    # plot training data
    plt.figure(1)
    plot_data(X, t)
    plt.savefig("graph/training_data.png")
    print('saved training data image as "training_data.png"')

    # create model
    model = Sequential([
        Dense(1, input_shape=(2, )),
        Activation('sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    plot = PlotEveryEpoch(X)
    #model.fit(X, t, nb_epoch=200, batch_size=5, verbose=2, callbacks=[plot])
    model.fit(X, t, nb_epoch=1000, batch_size=5, verbose=1)
    

if __name__ == '__main__':
    main()
