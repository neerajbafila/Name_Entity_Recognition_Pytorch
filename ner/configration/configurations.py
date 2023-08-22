from ner.utils.common import read_config, create_directories
from ner.constants import *
from ner.exception_and_logger.logger import logger
from ner.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig, DataValidationConfig, ModelTrainConfig
import os, sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

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

    def get_data_prepration_config(self) -> DataPreprocessingConfig:
        try:
            model_name = self.config[BASE_MODEL_CONFIG_KEY][BASE_MODEL_NAME_KEY]
            ner_tags = self.config[DATA_PREPROCESSING_KEY][NER_TAGS_KEY]

            index2tag = {idx:tag for idx, tag in enumerate(ner_tags)}
            tag2index = {tag:idx for idx, tag in enumerate(ner_tags)}
            data_store_path = self.config[PATH_KEY][DATA_STORE_KEY]
            data_store_full_path = Path(os.path.join(self.config[PATH_KEY][ARTIFACTS_KEY], data_store_path))
            self.my_logger.write_log(f"getting tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=data_store_full_path)
            data_prepration_config = DataPreprocessingConfig(
                model_name=model_name, index2tag=index2tag, tag2index=tag2index,
                tokenizer = tokenizer
            )
            self.my_logger.write_log(f"get_data_prepration_config done with below configuration\n {data_prepration_config}")
            
            return data_prepration_config

        except Exception as e:
            self.my_logger.write_exception(e)
    
    def get_data_validation_config(self) -> DataValidationConfig: 
        try:    
            self.my_logger.write_log(f"Getting data validation config")
            data_split = self.config[DATA_VALIDATION_CONFIG_KEY][DATA_SPLIT_KEY]
            columns_check = self.config[DATA_VALIDATION_CONFIG_KEY][COLUMNS_CHECKS_KEY]
            type_check = self.config[DATA_VALIDATION_CONFIG_KEY][TYPE_CHECK_KEY]
            data_validation_config = DataValidationConfig(dataset=None, data_split=data_split, columns_check=columns_check,
                                                          type_check=type_check)
            return data_validation_config
        except Exception as e:
            self.my_logger.write_exception(e)
    
    def get_model_training_config(self)->ModelTrainConfig:

        try:
            self.my_logger.write_log(f"getting model_training_config")
            model_name = self.config[BASE_MODEL_CONFIG_KEY][BASE_MODEL_NAME_KEY]
            num_class = self.config[BASE_MODEL_CONFIG_KEY][NUM_CLASS_KEY]
            num_epochs = self.config[BASE_MODEL_CONFIG_KEY][NUM_EPOCHS_KEY]
            batch_size = self.config[BASE_MODEL_CONFIG_KEY][BATCH_SIZE_KEY]
            save_steps = self.config[BASE_MODEL_CONFIG_KEY][SAVE_STEPS_KEY]
            ner_tags = self.config[DATA_PREPROCESSING_KEY][NER_TAGS_KEY]
            index2tag = {idx: tag for idx, tag in enumerate(ner_tags)}
            tag2index = {tag: idx for idx, tag in enumerate(ner_tags)}
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            output_root_dir = self.config[PATH_KEY][ARTIFACTS_KEY]
            output_dir = os.path.join(output_root_dir, self.config[PATH_KEY][OUTPUT_DIR_NAME_KEY])
            xlmr_config = AutoConfig.from_pretrained(model_name, num_labels=num_class, id2label=index2tag,
                                                     label2id=tag2index, cache_dir=output_dir)
            
            model_train_config = ModelTrainConfig(model_name=model_name, num_classes=num_class, num_epochs=num_epochs,
                                                  batch_size=batch_size, save_steps=save_steps, index2tag=index2tag,
                                                  tag2index=tag2index, tokenizer=tokenizer, xlmr_config=xlmr_config, output_dir=output_dir)
            
            return model_train_config
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)




# ob = Configuration()
# # ob.get_data_prepration_config()
# ob.get_data_validation_config()
# ob.get_model_training_config()
