import cv2
import os
import keras
import numpy as np

from pycocotools import mask
from pycocotools.coco import COCO


class Evaluator:
    """Pipeline for evaluating different metrics.

    # Arguments:
        data_path: str
            absolute path to validation directory
        model: keras.Model
            trained model
        random_state: np.random.RandomState
            random_state which will be used for drawing images from the validation data
    """

    def __init__(self: 'Evaluator',
                 data_path: str,
                 model: keras.Model,
                 random_state: np.random.RandomState):
        self.images_path = os.path.join(data_path, 'images')
        self.annotations_path = os.path.join(data_path, 'annotation-small.json')
        self.coco = COCO(self.annotations_path)

        self.model = model

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

    def evaluate(self):
        # TODO
        print('Not implemented yet!')

    def load_random_batch(self, batch_size: int = 16):
        """Loads a random batch of coco annotated images.

        # Arguments:
            batch_size: int
                Amount of images to draw

        # Returns:
            array of img objects
        """
        image_ids = self.coco.getImgIds()
        random_ids = self.random_state.choice(image_ids, batch_size).tolist()
        images = self.coco.loadImgs(random_ids)
        return images

    def get_random_prediction(self, batch_size: int = 4):
        """Gets predictions for <batch_size> random images.

        # Arguments:
            batch_size: int
                amount of images to draw

        # Returns
            Tuple of images (3-channeled), truth and prediction (both 1-channeled)
        """
        image_annotations = self.load_random_batch(batch_size)

        images, truth, predictions = [], [], []
        for image_annotation in image_annotations:
            image = self._image_as_array(image_annotation)
            label = self._label_as_array(image_annotation)
            prediction = self._prediction_as_array(image)

            images.append(image)
            truth.append(label)
            predictions.append(prediction)

        return images, truth, predictions

    def _image_as_array(self, image_annotation):
        path = os.path.join(self.images_path, image_annotation['file_name'])
        img = cv2.imread(path)
        return self._down_sample(img)

    def _label_as_array(self, image_annotation):
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_annotation['id']))
        return self._mask_from_annotation(annotations, (image_annotation['height'], image_annotation['width']))

    def _prediction_as_array(self, image):
        image = image.reshape((1, *image.shape)) / 255
        prediction = self.model.predict(image)
        return np.squeeze(prediction)

    def _down_sample(self, image):
        # shape: (height, width, channels)
        image = np.delete(image, list(range(1, image.shape[0], 2)), axis=0)
        image = np.delete(image, list(range(1, image.shape[1], 2)), axis=1)
        return image

    def _mask_from_annotation(self, annotation, shape):
        label = np.zeros(shape)
        for a in annotation:
            rle = mask.frPyObjects(a['segmentation'], *shape)
            m = np.squeeze(mask.decode(rle))
            label = np.logical_or(label, m)
        return self._down_sample(label)
