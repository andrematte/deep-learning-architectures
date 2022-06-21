# GoogLeNet Inception Architeture
# Author: Christian Szegedy, 2015
# Paper link: http://ieeexplore.ieee.org/document/7298594/

from tensorflow.keras.initializers import Constant, glorot_uniform
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model


def inception_module(
    x,
    filters_1x1,
    filters_3x3_reduce,
    filters_3x3,
    filters_5x5_reduce,
    filters_5x5,
    filters_pool_proj,
    bias_init,
    kernel_init,
    name=None,
):

    # Route 1: 1x1 Convolution
    conv_1x1 = Conv2D(
        filters_1x1,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(x)

    # Route 2: 1x1 Convolution + 3x3 Convolution
    pre_conv_3x3 = Conv2D(
        filters_3x3_reduce,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(x)

    conv_3x3 = Conv2D(
        filters_3x3,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(pre_conv_3x3)

    # Route 3: 1x1 Convolution + 5x5 Convolution
    pre_conv_5x5 = Conv2D(
        filters_5x5_reduce,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(x)

    conv_5x5 = Conv2D(
        filters_5x5,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(pre_conv_5x5)

    # Route 4: 3x3 Max Pooling + 1x1 Convolution
    pool_proj = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)

    pool_proj = Conv2D(
        filters_pool_proj,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(pool_proj)

    # Concatenation of the output of every route
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output


def googlenet():

    # Weight Initializer
    kernel_init = glorot_uniform()
    bias_init = Constant(value=0.2)

    # Input Layer (224,224,3)
    input_layer = Input(shape=(224, 224, 3))

    # GoogLeNet Part 1
    x = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding="same",
        strides=(2, 2),
        activation="relu",
        name="conv_1_7x7/2",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(input_layer)

    x = MaxPooling2D(
        pool_size=(3, 3), padding="same", strides=(2, 2), name="max_pool_1_3x3/2"
    )(x)

    x = BatchNormalization()(x)

    x = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        strides=(1, 1),
        activation="relu",
        name="conv_2a_1x1/1",
    )(x)

    x = Conv2D(
        filters=192,
        kernel_size=(3, 3),
        padding="same",
        strides=(1, 1),
        activation="relu",
        name="conv_2b_3x3/1",
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        pool_size=(3, 3), padding="same", strides=(2, 2), name="max_pool_2_3x3/2"
    )(x)

    # GoogLeNet Part 2
    x = inception_module(
        x,
        filters_1x1=64,
        filters_3x3_reduce=96,
        filters_3x3=128,
        filters_5x5_reduce=16,
        filters_5x5=32,
        filters_pool_proj=32,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_3a",
    )

    x = inception_module(
        x,
        filters_1x1=128,
        filters_3x3_reduce=128,
        filters_3x3=192,
        filters_5x5_reduce=32,
        filters_5x5=96,
        filters_pool_proj=64,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_3b",
    )

    x = MaxPooling2D(
        pool_size=(3, 3), padding="same", strides=(2, 2), name="max_pool_3_3x3/2"
    )(x)

    x = inception_module(
        x,
        filters_1x1=192,
        filters_3x3_reduce=96,
        filters_3x3=208,
        filters_5x5_reduce=16,
        filters_5x5=48,
        filters_pool_proj=64,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_4a",
    )

    x = inception_module(
        x,
        filters_1x1=160,
        filters_3x3_reduce=112,
        filters_3x3=224,
        filters_5x5_reduce=24,
        filters_5x5=64,
        filters_pool_proj=64,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_4b",
    )

    x = inception_module(
        x,
        filters_1x1=128,
        filters_3x3_reduce=128,
        filters_3x3=256,
        filters_5x5_reduce=24,
        filters_5x5=64,
        filters_pool_proj=64,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_4c",
    )

    x = inception_module(
        x,
        filters_1x1=112,
        filters_3x3_reduce=144,
        filters_3x3=288,
        filters_5x5_reduce=32,
        filters_5x5=64,
        filters_pool_proj=64,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_4d",
    )

    x = inception_module(
        x,
        filters_1x1=256,
        filters_3x3_reduce=160,
        filters_3x3=320,
        filters_5x5_reduce=32,
        filters_5x5=128,
        filters_pool_proj=128,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_4e",
    )

    x = MaxPooling2D(
        pool_size=(3, 3), padding="same", strides=(2, 2), name="max_pool_4_3x3/2"
    )(x)

    x = inception_module(
        x,
        filters_1x1=256,
        filters_3x3_reduce=160,
        filters_3x3=320,
        filters_5x5_reduce=32,
        filters_5x5=128,
        filters_pool_proj=128,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_5a",
    )

    x = inception_module(
        x,
        filters_1x1=384,
        filters_3x3_reduce=192,
        filters_3x3=384,
        filters_5x5_reduce=48,
        filters_5x5=128,
        filters_pool_proj=128,
        kernel_init=kernel_init,
        bias_init=bias_init,
        name="inception_5b",
    )

    # GoogLeNet Part 3

    x = AveragePooling2D(
        pool_size=(7, 7), padding="valid", strides=1, name="avg_pool_1_7x7/1"
    )(x)

    x = Dropout(0.4)(x)

    x = Dense(units=1000, activation="relu", name="linear_1")(x)

    x = Dense(units=1000, activation="relu", name="softmax_output")(x)

    model = Model(input_layer, [x], name="GoogLeNet")

    return model


def main():
    model = googlenet()
    print(model.summary())


if __name__ == "__main__":
    main()
