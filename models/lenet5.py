# LeNet-5 Architeture
# Author: Yann Lecun, 1998
# Paper link: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential


def lenet5():
    model = Sequential()

    # Layer 1: Convolution (Input Shape is 28x28x1)
    model.add(Conv2D(filters=6, kernel_size=5, strides=1,
              activation='tanh', input_shape=(28, 28, 1), padding='same'))

    # Subsampling (Average Pooling)
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))

    # Layer 2: Convolution
    model.add(Conv2D(filters=16, kernel_size=5, strides=1,
              activation='tanh', padding='valid'))

    # Subsampling (Average Pooling)
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))

    # Layer 3: Convolution
    model.add(Conv2D(filters=120, kernel_size=5, strides=1,
              activation='tanh', padding='valid'))

    # Flatten for classification
    model.add(Flatten())

    # Layer 4: Fully-connected
    model.add(Dense(units=84, activation='tanh'))

    # Layer 5: Fully-connected (Output Shape: 10)
    model.add(Dense(units=10, activation='softmax'))

    return model


def main():
    model = lenet5()
    print(model.summary())


if __name__ == '__main__':
    main()
