from ner.exception_and_logger.logger import logger
from ner.constants import *
from ner.entity.config_entity import ModelTrainConfig
from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.configration.configurations import Configuration
from typing import List, Dict, Any

class TrainTokenClassifier:preprocessed
    def __init__(self, model_train_config: ModelTrainConfig, prepocessed_data:Dict):
        self.my_logger = logger(CONFIG_FILE_NAME)
        self.model_train_config = model_train_config
        self.prepocessed_data = prepocessed_data
    

if __name__ == '__main__':
    conf_ob = Connfigration()


