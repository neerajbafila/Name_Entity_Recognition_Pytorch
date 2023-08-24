from ner.entity.config_entity import DataValidationConfig
from ner.exception_and_logger.logger import logger
from ner.configration.configurations import Configuration
from ner.components.data_ingestion import DataIngestion
from ner.constants import *
from typing import List
import pandas as pd
import sys

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data: dict):
        self.my_logger = logger(CONFIG_FILE_NAME)
        self.data_validation_config =  data_validation_config
        self.data = data
    
    def check_columns_names(self) -> bool:
        try:
            self.my_logger.write_log(f"columns validation started")
            split_name = self.data_validation_config.data_split
            columns_names = DataValidationConfig.columns_check
            columns_checks = []
            for split_n in split_name:
                # print(pd.DataFrame(self.data[split_n]).columns)
                columns_checks.append(pd.DataFrame(self.data[split_n]).columns == self.data_validation_config.columns_check) # validate the name
            columns_checks = sum(columns_checks) # [3 3 3]

            if sum(columns_checks) == len(self.data_validation_config.data_split) * len(self.data_validation_config.columns_check):
                # print(len(self.data_validation_config.data_split) * len(self.data_validation_config.columns_check)) # 9 = 3 * 3
                self.my_logger.write_log(f"columns validation completed ")
                return True
            else:
                return False
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)

    def type_check(self):
        try:
            self.my_logger.write_log(f"data type checking started")
            types = self.data_validation_config.type_check
            split_name = self.data_validation_config.data_split
            checks = False
            for split_n in split_name:
                if self.data[split_n].features['tokens'].feature.dtype == types[0] and self.data[split_n].features['ner_tags'].feature.dtype == types[1] and self.data[split_n].features['langs'].feature.dtype == types[2]:
                    checks = True
                else:
                    checks = False
            self.my_logger.write_log(f"data type checking completed")
            return checks
        
        except Exception as e:
            self.my_logger.write_exception(e)
        
    def drive_check(self) -> List[List[bool]]:
        try:
            self.my_logger.write_log(f"checks initiated")
            checks = list()
            checks.append([
                self.check_columns_names(),
                self.type_check()
            ])
        
            self.my_logger.write_log(f"checks completed Result \n {checks}")
            return checks
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())
        
        

            
if __name__ == '__main__':
    ob_c = Configuration()
    ings = DataIngestion(ob_c.get_data_ingestion_config())
    ob = DataValidation(ob_c.get_data_validation_config(), ings.get_data())
    # ob.check_columns_names()
    c = ob.drive_check()
    print(c)
    print(sum(c[0]))