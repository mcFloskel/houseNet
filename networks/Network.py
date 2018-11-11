import configparser

from keras.metrics import binary_accuracy
from keras.optimizers import Adam

from util.augmentation import NumpyDataLoader
from util.callbacks import setup_callbacks
from util.losses import dice
from util.metrics import intersection_over_union
from util.tf_utils import start_session


class Network:
    """Base class for the different network models.
    """

    def __init__(self: 'Network'):
        self.model = None

    def compile(self: 'Network'):
        self.model.compile(optimizer=Adam(), loss=dice, metrics=[intersection_over_union, binary_accuracy])
        self.model.summary()

    def train(self: 'Network',
              path_config: configparser.ConfigParser,
              weights_file_name: str,
              pre_trained_model_file_name: str = '',
              batch_size: int = 32,
              epochs: int = 50,
              random_state: int = None,
              checkpoint_period: int = 10,
              verbose: int = 1,
              use_gpu: bool = True):
        """Trains the model.

        # Arguments:
            path_config: configParser.ConfigParser
                configuration which contains absolute paths to train_data, val_data, models, logs
            weights_file_name: string
                filename for the saved weights
            pre_trained_model_file_name: string
                absolute path to an already trained model
            batch_size: integer
                amount of input images which are processed as a batch during learning
            epochs: integer
                Total amount of epochs which should be trained
            random_state: int of numpy.random.RandomState
                random state which should be used for data augmentation
            checkpoint_period: int
                period in which the model gets saved
            verbose: int
                verbosity level (0,1 or 2)
            use_gpu: boolean
                GPU is used for training (CPU will be used for data loading/augmentation)
        """

        if not self._is_model_and_config_correct(path_config):
            return

        print('Setting up data loader ...')
        train_data_loader = NumpyDataLoader(data_directory=path_config['DIRECTORIES']['train_data'],
                                            batch_size=batch_size,
                                            random_state=random_state)

        val_data_loader = NumpyDataLoader(data_directory=path_config['DIRECTORIES']['val_data'],
                                          batch_size=batch_size,
                                          shuffle=False,
                                          augment=False,
                                          random_state=random_state)

        print('Setting up callbacks ...')
        callbacks = setup_callbacks(path_config, weights_file_name, checkpoint_period)

        print('Starting session ...')
        start_session(use_gpu)
        if pre_trained_model_file_name != '':
            print('Loading pre-trained model ...')
            self.model.load_weights(pre_trained_model_file_name)

        print('Start training ...')
        self.model.fit_generator(generator=train_data_loader,
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 validation_data=val_data_loader,
                                 workers=6,
                                 use_multiprocessing=True)

    def _is_model_and_config_correct(self, path_config):
        if self.model is None:
            print('No model exists! Extend this class and set self.model to a keras model.')
            return False
        if type(path_config) is not configparser.ConfigParser:
            print('Path configuration is wrong! Use configparser.ConfigParser to read the configuration.')
            return False
        return True
