import configparser
import cv2
import os
import numpy as np

from keras.engine.saving import load_model

from analysis.evaluation import Evaluator
from util.losses import dice
from util.metrics import intersection_over_union


def visualize(images, ground_truth, predictions):
    ground_truth = [cv2.cvtColor(gt.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for gt in ground_truth]
    predictions = [cv2.cvtColor(p.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for p in predictions]

    collection = np.ones((750, 550, 3), dtype=np.uint8) * 50
    for i in range(len(images)):
        pos = i * 200
        collection[pos: pos + 150, 0:150, :] = images[i]
        collection[pos: pos + 150, 200:350, :] = ground_truth[i]
        collection[pos: pos + 150, 400:550, :] = predictions[i]
    cv2.imshow('Example Prediction', collection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Get config
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.pardir, 'config.ini'))

    # Load model
    PATH_TO_MODEL = os.path.join(config['DIRECTORIES']['models'], 'my_rNet.hdf5')
    custom_objects = {'dice': dice, 'intersection_over_union': intersection_over_union}
    my_model = load_model(PATH_TO_MODEL, custom_objects=custom_objects)

    # Evaluate and visualize
    PATH_TO_DATA = config['DIRECTORIES']['val_data']
    evaluator = Evaluator(PATH_TO_DATA, my_model, np.random.RandomState(2013))
    data = evaluator.get_random_prediction()
    visualize(*data)
