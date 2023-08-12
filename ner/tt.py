# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name())
# print(torch.__version__)
# print(torch.version.cuda)

# from transformers import pipeline



# import tensorflow as tf
# print(tf.config.list_physical_devices())

from constants import *
from ner.utils.common import read_config, create_directories

content = read_config(config_file=CONFIG_FILE_NAME)
create_directories(content['paths']['logs'])
print(content)