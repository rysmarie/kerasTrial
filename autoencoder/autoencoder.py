import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K


def plotimg(original, decoded, n=10):
    plt.figure(figsize=(20, 4))

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('result.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='autoencoder')
    parser.add_argument('--dim', type=int, default=32,
                        help='encoding dimention')
    args = parser.parse_args()

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # x_train[1:] = [28, 28]
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    input_img = Input(shape=(784, ))
    encoded = Dense(args.dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)

    encoded_input = Input((args.dim, ))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=256,
                    shuffle=True, validation_data=(x_test, x_test), verbose=1)

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    plotimg(x_test, decoded_imgs, 10)

    print('finished')

    # skip making encoded and decoded images

if __name__ == '__main__':
    main()
