import os

from keras.utils import Sequence
import numpy as np


class NumpyDataLoader(Sequence):
    """
    Loads data/labels where each data point is stored as a numpy array (.npy) on the file system.
    The data points/labels have to be stored with a given prefix and incrementing numbers for
    identification, for example with the prefix 'x_':
    x_1.npy, x_2.npy, ...

    # Arguments:
        data_directory: string
            absolute path to the directory
        batch_size: int
            amount of data points which are processed as one batch
        data_shape: tuple
            shape of the data
        labels_shape: tuple
            shape  of the labels
        data_prefix: string
            prefix for the data files
        label_prefix: string
            prefix for the label files
        shuffle: boolean
            Shuffle data points after each epoch
        augment: boolean
            augment the data points with random horizontal/vertical flips and 90° rotations
        random_state: int or numpy.random.RandomState
            state which will be used for shuffling and augmentation
    """

    def __init__(self,
                 data_directory: str,
                 batch_size: int = 32,
                 data_prefix: str = 'x_',
                 label_prefix: str = 'y_',
                 shuffle: bool = True,
                 augment: bool = True,
                 random_state: np.random.RandomState = None):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.data_prefix = data_prefix
        self.label_prefix = label_prefix
        self.shuffle = shuffle
        self.augment = augment

        # Get amount of data points
        data_size = len([f for f in os.listdir(data_directory) if f.startswith(data_prefix)])
        self.image_ids = np.arange(data_size)

        # Get data/label dimensionality from first sample
        # Currently only one channeled labels are supported (binary segmentation)
        # Augmentation might crash when label shape is not 2d
        self.data_shape = np.load(data_directory + data_prefix + '0.npy').shape
        self.labels_shape = (*np.load(data_directory + label_prefix + '0.npy').shape, 1)

        # Init random state
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]

        data = np.empty((self.batch_size, *self.data_shape))
        labels = np.empty((self.batch_size, *self.labels_shape))

        for i, ID in enumerate(indexes):
            x = np.load(self.data_directory + 'x_' + str(ID) + '.npy') / 255
            y = np.load(self.data_directory + 'y_' + str(ID) + '.npy').reshape(self.labels_shape)

            if self.augment:
                rotations = self.random_state.randint(4)
                flip_lr = self.random_state.randint(2)
                flip_ud = self.random_state.randint(2)

                if flip_lr:
                    x = np.fliplr(x)
                    y = np.fliplr(y)

                if flip_ud:
                    x = np.flipud(x)
                    y = np.flipud(y)

                x = np.rot90(x, k=rotations)
                y = np.rot90(y, k=rotations)

            data[i,] = x
            labels[i,] = y

        return data, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state.shuffle(self.image_ids)
