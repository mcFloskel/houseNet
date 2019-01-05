import configparser
import json
import os

from keras import Model
from keras.engine.saving import load_model
from keras.layers import UpSampling2D

from analysis.evaluation import Evaluator

from util.losses import dice
from util.metrics import intersection_over_union

# Get config
config = configparser.ConfigParser()
config.read(os.path.join(os.path.pardir, 'config.ini'))

# Build all paths
path_to_model = os.path.join(config['DIRECTORIES']['models'], 'my_rNet.hdf5')
path_to_predictions = os.path.join(config['DIRECTORIES']['predictions'], 'rNet_prediction.json')
path_to_dataset = config['DIRECTORIES']['val_data']

# Load model
custom_objects = {'dice': dice, 'intersection_over_union': intersection_over_union}
my_model = load_model(path_to_model, custom_objects=custom_objects)

# Up sample model output -> model was trained on images with shape (150, 150)
up_sampler = UpSampling2D()(my_model.output)
model_wrapper = Model(my_model.input, up_sampler)


# Initialize evaluator
evaluator = Evaluator(path_to_dataset, model_wrapper)

# Can be skipped if you have already saved the predictions
# evaluator.save_predictions_as_json(path_to_predictions)

# Evaluate
prediction_annotations = json.loads(open(path_to_predictions).read())
evaluator.evaluate(prediction_annotations)
