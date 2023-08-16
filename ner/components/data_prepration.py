from ner.exception_and_logger.logger import logger
from ner.configration.configurations import Configuration
from ner.entity.config_entity import DataPreprocessingConfig
from ner.constants import *
import sys
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
    
    def tokenize_and_align_labels(self, data):
        try:

            tokenizer = self.data_preprocessing_config.tokenizer
            tokenize_input = tokenizer(data['tokens'], truncation=True, is_split_into_words=True)
            labels = []
            for idx, label in enumerate(data['ner_tags']):
                words_id = tokenize_input.word_ids(batch_index=idx)
                previous_word_idx = None
                label_ids = []
                for word_idx in words_id:
                    if word_idx is None or word_idx == previous_word_idx:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenize_input['labels'] = labels
            return tokenize_input
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    
    def encode_en_dataset(self, corpus) -> dict:
        try:
            self.my_logger.write_log(f"encoding dataset")
            return corpus.map(self.tokenize_and_align_labels, batched=True, remove_columns=['langs', 'ner_tags', 'tokens'])
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    
    def prepare_data_for_fine_tuning(self) -> dict:
        try:
            # create _tags
            self.my_logger.write_log(f"Creating NER Tags")
            self.data = self.data.map(self.create_tag_name)
            self.my_logger.write_log(f"Creating NER Tags completed successfully")
            # encode data with label
            self.my_logger.write_log(f"Tokenizing and aligning labels")
            panx_en_encoded_data = self.encode_en_dataset(self.data)
            self.my_logger.write_log(f"Tokenizing and aligning labels completed successfully")

            return panx_en_encoded_data
        except Exception as e:
            self.my_logger.write_exception(e)

                
if __name__ == '__main__':
    config = Configuration()
    
    ob = DataIngestion(config.get_data_ingestion_config())
    d = ob.get_data()
    data_prep = DataPreprocessing(config.get_data_prepration_config(), d)
    encoded_data = data_prep.prepare_data_for_fine_tuning()
    print(encoded_data)
    # print(encoded_data['train']['input_ids'][:10])
    # d = d.map(data_prep.create_tag_name)
    # print(d)
    # d.map(data_prep.tokenize_and_align_labels, batched=True)