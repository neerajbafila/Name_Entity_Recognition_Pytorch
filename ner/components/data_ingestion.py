from ner.constants import *
from ner.entity.config_entity import DataIngestionConfig
from ner.configration.configurations import Configuration
from ner.exception_and_logger.logger import logger
from datasets import load_dataset

STAGE = "DataIngestion"

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
          self.my_logger = logger(CONFIG_FILE_NAME)
          self.my_logger.write_log(f"{STAGE} stated")
          self.data_ingestion_config = data_ingestion_config
    
    def get_data(self):
         
         try:  
            self.my_logger.write_log(f"getting data from Huggingface")
            pan_x_en_data = load_dataset(self.data_ingestion_config.dataset_name,
                                    name=self.data_ingestion_config.subset_name,
                                    cache_dir=self.data_ingestion_config.data_path)
            
            self.my_logger.write_log(f"Dataset Info : {pan_x_en_data}")
            return pan_x_en_data
         
         except Exception as e:
              self.my_logger.write_exception(e)

        
if __name__ == "__main__":
     config_ob = Configuration()
     data_ingestion_ob = DataIngestion(config_ob.get_data_ingestion_config())
     data_ingestion_ob.get_data()
              
         


