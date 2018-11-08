import configparser

from networks.UNet_3 import UNet3

config = configparser.ConfigParser()
config.read('config.ini')
weights_file_name = 'my_unet'

net = UNet3()
net.train(path_config=config,
          weights_file_name=weights_file_name,
          batch_size=8,
          random_state=2009,
          epochs=50,
          checkpoint_period=10,
          verbose=2)
