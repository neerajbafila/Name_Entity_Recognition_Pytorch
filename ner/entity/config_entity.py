from collections import namedtuple

DataIngestionConfig = namedtuple('DataIngestionConfig', ["dataset_name", "subset_name", "data_path"])
DataPreprocessingConfig = namedtuple('DataPreprocessingConfig', ["model_name","index2tag", "tag2index", "tokenizer"])
DataValidationConfig = namedtuple("DataValidationConfig", ["dataset", "data_split", "columns_check", "type_check"])
ModelTrainConfig = namedtuple("ModelTrainConfig", ["model_name", "num_classes", 
                                                   "num_epochs", "batch_size", "save_steps",
                                                   "index2tag", "tag2index",
                                                   "tokenizer", "xlmr_config", "output_dir"])
