import tensorflow as tf
import keras.backend as K


def start_session(use_gpu=True):
    """Sets tensorflow gpu options and initializes global/local variables
    """

    if use_gpu:
        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(session)
    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.local_variables_initializer())
