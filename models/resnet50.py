# ResNet Architeture
# Author: Kaiming He, 2015
# Paper link: https://arxiv.org/pdf/1512.03385.pdf

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def bottleneck_residual_block(X, middle_kernel, filters, reduce=False, s=2):

    # Unpack filters of each convolutional layer
    F1, F2, F3 = filters

    X_shortcut = X

    if reduce:
        X_shortcut = Conv2D(
            filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid"
        )(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        # First component with reduce
        X = Conv2D(F1, (1, 1), strides=(s, s), padding="valid")(X)

    else:
        # First Component without reduce
        X = Conv2D(F1, (1, 1), strides=(1, 1), padding="valid")(X)

    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    # Second Component
    X = Conv2D(F2, middle_kernel, strides=(1, 1), padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    # Third Component
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid")(X)
    X = BatchNormalization(axis=3)(X)

    # Adding Residue before activation layer
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def resnet50(input_shape=(32, 32, 3), classes=10):
    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        name="conv1",
        kernel_initializer=glorot_uniform(seed=23),
    )(X_input)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = bottleneck_residual_block(X, 3, [64, 64, 256], reduce=True, s=1)
    X = bottleneck_residual_block(X, 3, [64, 64, 256])
    X = bottleneck_residual_block(X, 3, [64, 64, 256])

    # Stage 3
    X = bottleneck_residual_block(X, 3, [128, 128, 512], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [128, 128, 512])
    X = bottleneck_residual_block(X, 3, [128, 128, 512])

    # Stage 4
    X = bottleneck_residual_block(X, 3, [256, 256, 1024], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])
    X = bottleneck_residual_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = bottleneck_residual_block(X, 3, [512, 512, 2048], reduce=True, s=2)
    X = bottleneck_residual_block(X, 3, [512, 512, 2048])
    X = bottleneck_residual_block(X, 3, [512, 512, 2048])

    # Pooling Layer
    X = AveragePooling2D(pool_size=(1, 1))(X)

    # Classification Layer
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc_" + str(classes))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model


def main():
    model = resnet50()
    print(model.summary())


if __name__ == "__main__":
    main()
