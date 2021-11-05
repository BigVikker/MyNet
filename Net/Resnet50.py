from keras.regularizers import l2
from tensorflow import keras

def resnet50(nb_classes=7, num_filters=64, drop_out=0.25):
    inputs = keras.layers.Input(shape=(48, 48, 1))

    # frist conv
    x = keras.layers.Conv2D(num_filters, (2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(drop_out)(x)

    # block 1 # stack 1

    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)
    x = keras.layers.Conv2D(num_filters * 3, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)

    # combine stack

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 2
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 3
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # end block1
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    num_filters = num_filters + num_filters

    # block 2

    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)

    x = keras.layers.Conv2D(num_filters * 3, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)

    # combine stack

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 2
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 3
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    num_filters = num_filters + num_filters
    # block 3
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)

    x = keras.layers.Conv2D(num_filters * 3, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)

    # combine stack

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 2
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 3
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    num_filters = num_filters + num_filters

    # block 4

    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)

    x = keras.layers.Conv2D(num_filters * 3, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)

    # combine stack

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 2
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # stack 3
    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    y = keras.layers.Conv2D(num_filters, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 3, (2, 2), use_bias=False, padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)

    x = keras.layers.Conv2D(num_filters * 3, (3, 3), use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)

    # combine stack

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    y = keras.layers.Flatten()(x)
    y = keras.layers.Dense(1024)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(.5)(y)  # .5
    y = keras.layers.Dense(1024)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(.5)(y)  # .5
    outputs = keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model


