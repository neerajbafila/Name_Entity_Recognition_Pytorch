from collections import namedtuple

DataIngestionConfig = namedtuple('DataIngestionConfig', ["dataset_name", "subset_name", "data_path"])
DataPreprocessingConfig = namedtuple('DataPreprocessingConfig', ["model_name","index2tag", "tag2index", "tokenizer"])
