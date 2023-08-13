from ner.utils.common import read_config, create_directories
from ner.constants import *
from ner.exception_and_logger.logger import logger
from ner.entity.config_entity import DataIngestionConfig
import os
from pathlib import Path

class Configuration:
    def __init__(self):
        self.my_logger = logger(config_file=CONFIG_FILE_NAME)
        try:
            
            self.my_logger.write_log(f"Reading configuration file")
            self.config = read_config(config_file=CONFIG_FILE_NAME)
        except Exception as e:
            self.my_logger.write_exception(e)
            print(e)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        self.my_logger.write_log(f"Getting data ingestion config details")
        try:
            dataset_name = self.config[DATA_INGESTION_CONFIG_KEY][DATASET_NAME_KEY]
            subset_name = self.config[DATA_INGESTION_CONFIG_KEY][SUBSET_NAME_KEY]
            data_store_path = self.config[PATH_KEY][DATA_STORE_KEY]
            data_store_full_path = Path(os.path.join(self.config[PATH_KEY][ARTIFACTS_KEY], data_store_path))

            data_ingestion_config = DataIngestionConfig(dataset_name=dataset_name, subset_name=subset_name,
                                                        data_path=data_store_full_path)
            
            self.my_logger.write_log(f"Below are the dataset and data store details \n {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            self.my_logger.write_exception(e)
