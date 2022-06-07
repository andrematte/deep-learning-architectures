# AlexNet Architeture
# Author: Alex Krizhevsky, 2017
# Paper link: https://dl.acm.org/doi/10.1145/3065386

from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def alexnet():

    model = Sequential()

    # Layer 1: Conv
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
              activation='relu', input_shape=(227, 227, 3), padding='valid'))

    # Batch Normalization and Subsampling (Max Pooling)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Layer 2: Conv
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
              activation='relu', padding='same', kernel_regularizer=l2(0.0005)))

    # Batch Normalization and Subsampling (Max Pooling)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # Layer 3: Conv
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
              activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Layer 4: Conv
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
              activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Layer 5: Conv
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
              activation='relu', padding='same', kernel_regularizer=l2(0.0005)))

    # Batch Normalization and Subsampling (Max Pooling)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Flatten
    model.add(Flatten())

    # Layer 6: Fully Connected + Dropout
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 7: Fully Connected + Dropout
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 8: Fully Connected (Output shape: 1000)
    model.add(Dense(units=1000, activation='softmax'))

    return model


def main():
    model = alexnet()
    print(model.summary())


if __name__ == '__main__':
    main()
