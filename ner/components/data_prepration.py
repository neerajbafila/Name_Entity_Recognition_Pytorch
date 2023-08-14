from ner.exception_and_logger.logger import logger
from ner.configration.configurations import Configuration
from ner.entity.config_entity import DataPreprocessingConfig
from ner.constants import *
from typing import Any
from ner.components.data_ingestion import DataIngestion
class DataPreprocessing:
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig, data: Any):
        self.my_logger = logger(CONFIG_FILE_NAME)
        self.data_preprocessing_config = data_preprocessing_config
        self.data = data
    
    def create_tag_name(self, data):
        try:
            return {'ner_tags_str': [self.data_preprocessing_config.index2tag[idx_no] for idx_no in data['ner_tags']]}
        except  Exception as e:
            self.my_logger.write_exception(e)
    

if __name__ == '__main__':
    config = Configuration()
    
    ob = DataIngestion(config.get_data_ingestion_config())
    d = ob.get_data()
    data_prep = DataPreprocessing(config.get_data_prepration_config(), d)
    d.map(data_prep.create_tag_name)