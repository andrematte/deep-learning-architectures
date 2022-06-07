# VGG-16 Architeture
# Author: Karen Simonyan, 2015
# Paper link: http://arxiv.org/abs/1409.1556

from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def vgg16():

    model = Sequential()

    # Convolution Block 1
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu',
              input_shape=(224, 224, 3)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution Block 2
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution Block 3
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution Block 4
    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution Block 5
    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3),
              strides=(1, 1), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Classification Block
    model.add(Flatten())

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1000, activation='softmax'))

    return model


def main():
    model = vgg16()
    print(model.summary())


if __name__ == '__main__':
    main()
