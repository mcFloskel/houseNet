from keras import Input, Model
from keras.layers import MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, concatenate, ZeroPadding2D

from networks.Network import Network


class UNet3(Network):
    """ UNet-like architecture with three pooling steps.
    This particular architecture was designed for processing data with
    (150, 150, 3) input shape and (150, 150, 1) output shape.
    """

    def __init__(self):
        super().__init__()

        # Down
        # Top
        input_layer = Input(shape=(150, 150, 3), name='input_layer')
        input_padded = ZeroPadding2D(name='input_padded')(input_layer)
        conv_top_1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='conv_top_1')(
            input_padded)
        conv_top_1_norm = BatchNormalization(name='conv_top_1_norm')(conv_top_1)
        conv_top_2 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='conv_top_2')(
            conv_top_1_norm)
        conv_top_2_norm = BatchNormalization(name='conv_top_2_norm')(conv_top_2)
        pool_top = MaxPooling2D(pool_size=2, name='pool_top')(conv_top_2_norm)

        # Middle
        conv_mid_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv_mid_1')(pool_top)
        conv_mid_1_norm = BatchNormalization(name='conv_mid_1_norm')(conv_mid_1)
        conv_mid_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv_mid_2')(
            conv_mid_1_norm)
        conv_mid_2_norm = BatchNormalization(name='conv_mid_2_norm')(conv_mid_2)
        pool_mid = MaxPooling2D(pool_size=2, name='pool_mid')(conv_mid_2_norm)

        # Bottom
        conv_bot_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv_bot_1')(pool_mid)
        conv_bot_1_norm = BatchNormalization(name='conv_bot_1_norm')(conv_bot_1)
        conv_bot_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv_bot_2')(
            conv_bot_1_norm)
        conv_bot_3_norm = BatchNormalization(name='conv_bot_3_norm')(conv_bot_2)
        pool_bot = MaxPooling2D(pool_size=2, name='pool_bot')(conv_bot_3_norm)

        # Vertical
        conv_vert_1 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='conv_vert_1')(
            pool_bot)
        conv_vert_1_norm = BatchNormalization(name='conv_vert_1_norm')(conv_vert_1)
        conv_vert_2 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='conv_vert_2')(
            conv_vert_1_norm)
        conv_vert_2_norm = BatchNormalization(name='conv_vert_2_norm')(conv_vert_2)

        # Up
        # Bottom
        vert_up = UpSampling2D(size=2, name='vert_up')(conv_vert_2_norm)
        conc_ver_bot = concatenate([vert_up, conv_bot_3_norm])
        conv_up_bot_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv_up_bot_1')(
            conc_ver_bot)
        conv_up_bot_1_norm = BatchNormalization(name='conv_up_bot_1_norm')(conv_up_bot_1)
        conv_up_bot_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv_up_bot_2')(
            conv_up_bot_1_norm)
        conv_up_bot_2_norm = BatchNormalization(name='conv_up_bot_2_norm')(conv_up_bot_2)

        # Middle
        bot_up = UpSampling2D(size=2, name='bot_up')(conv_up_bot_2_norm)
        conc_bot_mid = concatenate([bot_up, conv_mid_2_norm])
        conv_up_mid_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv_up_mid_1')(
            conc_bot_mid)
        conv_up_mid_1_norm = BatchNormalization(name='conv_up_mid_1_norm')(conv_up_mid_1)
        conv_up_mid_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv_up_mid_2')(
            conv_up_mid_1_norm)
        conv_up_mid_2_norm = BatchNormalization(name='conv_up_mid_2_norm')(conv_up_mid_2)

        # Top
        mid_up = UpSampling2D(size=2, name='mid_up')(conv_up_mid_2_norm)
        conc_mid_top = concatenate([mid_up, conv_top_2_norm])
        conv_up_top_1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='conv_up_top_1')(
            conc_mid_top)
        conv_up_top_1_norm = BatchNormalization(name='conv_up_top_1_norm')(conv_up_top_1)
        conv_up_top_2 = Conv2D(filters=32, kernel_size=3, activation='relu', name='conv_up_top_2')(
            conv_up_top_1_norm)
        conv_up_top_2_norm = BatchNormalization(name='conv_up_top_2_norm')(conv_up_top_2)

        output_layer = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='output_layer')(conv_up_top_2_norm)

        self.model = Model(input_layer, output_layer)
        self.compile()
