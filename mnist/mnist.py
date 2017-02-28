import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping


def build_multilayer_perceptron():
    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


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


def main():
    parser = argparse.ArgumentParser(description='mnist')
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    args = parser.parse_args()
    nb_classes = 10
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np_utils.to_categorical(y_train, nb_classes=10)
    Y_test = np_utils.to_categorical(y_test, nb_classes=10)

    # create model
    model = build_multilayer_perceptron()

    plot(model, show_shapes=True, show_layer_names=True, to_file='model.png')

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Early-stopping
    early_stopping = EarlyStopping(patience=0, verbose=1)

    # モデルの訓練
    history = model.fit(X_train, Y_train,
                        batch_size=args.batch,
                        nb_epoch=args.epoch,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    # plot_history(history)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test loss:', loss)
    print('Test acc:', acc)
    visualize_weight(model.layers[6].get_weights()[0], 'weights/weight')

if __name__ == '__main__':
    main()
