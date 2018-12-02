from keras.engine import Layer
from keras.layers import Concatenate

from networks.blocks.conv_block import convolution_block


def res_block(input_layer: Layer,
              filters: int,
              name: str,
              kernel_size: int = 3,
              activation: str = 'relu',
              padding: str = 'same',
              dilation_rate: int = 1):
    """Generate a residual block which also allows dilation.


    # Arguments:
        input_layer: Layer
            input for the first convolutional layer
        filters: integer
            amount of convolutional filters
        name: string
            prefix for the layer names
        kernel_size: integer
            size of the convolutional filter kernels, only quadratic kernels are allowed
        activation: string
            activation after each convolutional layer (see Keras documentation)
        padding: string
            padding for the input (either same or valid, see Keras documentation)
        dilation_rate: integer
            amount of dilation between filter weights

    # Returns:
        A Keras Layer
    """

    block = convolution_block(input_layer,
                              filters=filters,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding,
                              dilation_rate=dilation_rate,
                              name=name)
    return Concatenate(name=name + '_output')([input_layer, block])
