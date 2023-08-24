from ner.exception_and_logger.logger import logger
from ner.constants import *
from ner.components.data_ingestion import DataIngestion
from ner.components.data_validation import DataValidation
from ner.components.data_prepration import DataPreprocessing
from ner.components.model_training import TrainTokenClassifier
from ner.configration.configurations import Configuration
from typing import Dict, List
import sys

class TrainPipeline:
    def __init__(self,config):
        self.config = config
        self.my_logger = logger(CONFIG_FILE_NAME)
    
    def run_data_ingestion(self)->Dict:
        try:
            self.my_logger.write_log(f"running data_ingestion pipeline")
            data_ingestion = DataIngestion(self.config.get_data_ingestion_config())
            data = data_ingestion.get_data()
            return data
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())
    
    def run_data_validation(self, data)->List[List[bool]]:
        try:
            self.my_logger.write_log(f"running data validation pipeline")
            data_validation = DataValidation(self.config.get_data_validation_config(), data)
            checks = data_validation.drive_check()
            return checks
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())

    def run_data_preparation(self, data)->Dict:
        try:
            self.my_logger.write_log(f"Running Data Preparation pipeline")
            data_preparation = DataPreprocessing(self.config.get_data_prepration_config(), data)
            data = data_preparation.prepare_data_for_fine_tuning()
            return data
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())
    
    def run_model_training(self, data):
        try:
            self.my_logger.write_log(f"running model training pipeline")
            classifier = TrainTokenClassifier(self.config.get_model_training_config(), data)
            classifier.train()
            self.my_logger.write_log(f"training completed successfully") 
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())
    
    def run_pipeline(self):
        try:

            data = self.run_data_ingestion()
            checks = self.run_data_validation(data)
            if sum(checks[0]) == 2: # both checks should be True else sum !=2 
                self.my_logger.write_log(f"checks completed")
                prepared_data = self.run_data_preparation(data)
                self.my_logger.write_log(f"Preprocessed Data {prepared_data}")
                self.run_model_training(prepared_data)
            else:
                self.my_logger.write_log(f"checks failed {checks}")
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())

if __name__ == "__main__":
    # config_object = Configuration()
    pipeline = TrainPipeline(Configuration())
    pipeline.run_pipeline()

