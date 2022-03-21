from keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Activation, \
    Flatten, Dropout, BatchNormalization, MaxPool2D, multiply, Dense
from tensorflow.keras import Input

def vgg_resnet(w=256, h=256, c=1, drop_out=0.25, nout=2):
    input = Input(shape=(w,h,c), name='input')


    # vgg 1
    BN_1 = BatchNormalization(name='BN_1')(input)
    conv_1 = (Conv2D(32, (5, 5), kernel_regularizer=l2(1e-4), padding='same', activation='relu', name='conv_1'))(BN_1)  # , input_shape=(256, 256, 1)
    conv_2 = (Conv2D(32, (5, 5), padding='same', kernel_regularizer=l2(1e-4), activation='relu', name='conv_2'))(conv_1)
    max_pool_1 = (MaxPool2D((2, 2), name='max_pool_1'))(conv_2)
    BN_2 = (BatchNormalization(name='BN_2'))(max_pool_1)
    drop_out_1 = (Dropout(drop_out, name='drop_out_1'))(BN_2)
    #end block 1

    # resnet 1
    num_filters = 32
    y = keras.layers.Conv2D(num_filters, (2, 2), padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(drop_out_1)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 2, (2, 2), padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)
    x = keras.layers.Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(drop_out_1)
    x = keras.layers.BatchNormalization()(x)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)


    # vgg
    conv_3 = (Conv2D(64, (5, 5), padding='same', kernel_regularizer=l2(1e-4), activation='relu', name='conv_3'))(x)
    conv_4 = (Conv2D(64, (5, 5), padding='same', kernel_regularizer=l2(1e-4), activation='relu', name='conv_4'))(conv_3)
    max_pool_2 = (MaxPool2D((2, 2), name='max_pool_2'))(conv_4)
    BN_3 = (BatchNormalization(name='BN_3'))(max_pool_2)
    drop_out_2 = (Dropout(drop_out, name='drop_out_2'))(BN_3)

    # resnet 2
    num_filters = 64
    y = keras.layers.Conv2D(num_filters, (2, 2), padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(drop_out_2)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 2, (2, 2), padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)
    x = keras.layers.Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(drop_out_2)
    x = keras.layers.BatchNormalization()(x)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    # end block 2

    # vgg 3
    conv_5 = (Conv2D(128, (5, 5), padding='same', kernel_regularizer=l2(1e-4), activation='relu', name='conv_5'))(x)
    conv_6 = (Conv2D(128, (5, 5), padding='same', kernel_regularizer=l2(1e-4), activation='relu', name='conv_6'))(conv_5)
    max_pool_3 = (MaxPool2D((2, 2), name='max_pool_3'))(conv_6)
    BN_4 = (BatchNormalization(name='BN_4'))(max_pool_3)
    drop_out_3 = (Dropout(drop_out, name='drop_out_3'))(BN_4)
    # renet 3

    num_filters = 128
    y = keras.layers.Conv2D(num_filters, (2, 2), padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(drop_out_3)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters, (3, 3), kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.Dropout(drop_out)(y)

    y = keras.layers.Conv2D(num_filters * 2, (2, 2), padding='same', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(y)
    y = keras.layers.BatchNormalization()(y)
    x = keras.layers.Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(drop_out_3)
    x = keras.layers.BatchNormalization()(x)

    # combine stack
    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)
    #end_block 3
    drop_out_4 = (Dropout(drop_out, name='drop_out_4'))(x)
    flatten_1 = (Flatten(name='flatten_1'))(drop_out_4)
    dense_1 = Dense(128, activation='relu', name='dense_1', kernel_initializer='he_normal')(flatten_1)
    drop_out_6 = Dropout(0.5, name='drop_out_6')(dense_1)
    classify = Dense(nout, activation='softmax', name='classify', kernel_initializer='he_normal')(drop_out_6)

    model_vgg_resnet = keras.models.Model(inputs=input, outputs=classify)
    return model_vgg_resnet


model = vgg_resnet()