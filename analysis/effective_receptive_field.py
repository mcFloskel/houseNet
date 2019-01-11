import keras.backend as K
import tensorflow as tf
import numpy as np
from keras import Model


def randomize_weights(model):
    print('Randomizing ...')
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer = getattr(v_arg, 'initializer')
                initializer.run(session=session)


def get_effective_receptive_field(model: Model,
                                  runs: int = 20,
                                  randomize_model: bool = True):
    """Calculates the effective receptive field for a given neural network architecture.
    The values are normalized after calculation for better visualization.

    # Arguments:
        model: Keras model
            a keras model
        runs: int
            number of runs for averaging
        randomize_model:
            Randomizes the model weights for each run (set this to False for trained models)

    # Returns:
        A Numpy array which represents the receptive field
    """

    input_shape = (1, *model.input_shape[1:])
    output_shape = (1, *model.output_shape[1:])

    center_x = int(input_shape[1] / 2)
    center_y = int(input_shape[2] / 2)

    initial = np.zeros(output_shape, dtype=np.float32)
    initial[:, center_x, center_y, :] = 1
    gradients = tf.gradients(ys=model.output, xs=model.input, grad_ys=initial)

    session = K.get_session()

    receptive_field = np.zeros(input_shape[1:])
    print('Calculating gradient ...')
    for i in range(runs):
        if randomize_model:
            randomize_weights(model)
        grad = session.run(gradients, feed_dict={model.input.name: np.random.rand(*input_shape)})
        receptive_field += grad[0][0]
        print('Finished run %d of %d' % (i + 1, runs))

    # Normalize
    receptive_field = receptive_field / runs
    receptive_field[receptive_field < 0] = 0
    receptive_field /= np.ptp(receptive_field)
    return receptive_field
