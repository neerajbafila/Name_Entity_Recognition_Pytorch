from ner.exception_and_logger.logger import logger
from ner.constants import *
from ner.entity.config_entity import ModelTrainConfig
from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.configration.configurations import Configuration
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from seqeval.metrics import f1_score
import sys
import numpy as np
from ner.components.data_ingestion import DataIngestion
from ner.components.data_prepration import DataPreprocessing

from typing import List, Dict, Any

class TrainTokenClassifier:
    def __init__(self, model_train_config: ModelTrainConfig, preprocessed_data:Dict):
        self.my_logger = logger(CONFIG_FILE_NAME)
        self.model_train_config = model_train_config
        self.preprocessed_data = preprocessed_data
    
    def create_training_arguments(self):
        try:
            self.my_logger.write_log(f"Creating training arguments")
            logging_steps = len(self.preprocessed_data["train"].select(range(100))) // self.model_train_config.batch_size
            training_args = TrainingArguments(output_dir=self.model_train_config.output_dir,
                                              log_level="error", num_train_epochs=self.model_train_config.num_epochs,
                                              per_device_train_batch_size=self.model_train_config.batch_size,
                                              per_device_eval_batch_size=self.model_train_config.batch_size,
                                              save_steps=self.model_train_config.save_steps,
                                              weight_decay=0.01,
                                              disable_tqdm=False,
                                              logging_steps=logging_steps,
                                              push_to_hub=False)
            self.my_logger.write_log(f"below training arguments\n {training_args} ")
            return training_args
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    
    def data_collector(self):
        try:
            self.my_logger.write_log(f"making data collector")
            return DataCollatorForTokenClassification(self.model_train_config.tokenizer)
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    
    def model_init(self):
        try:
            self.my_logger.write_log(f"Initialing the model")
            return XLMRobertaForTokenClassification.from_pretrained(self.model_train_config.model_name,
                                                                    config=self.model_train_config.xlmr_config)
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    
    def align_predictions(self, predictions, label_ids):
        try:
            preds = np.argmax(predictions, axis=2)
            batch_size, seq_len = preds.shape
            actual_label, pred_label = [], []

            for batch_idx in range(batch_size):
                example_label, example_pred = [], []
                for seq_idx in range(seq_len):
                    # Ignore label IDs = -100
                    if label_ids[batch_idx, seq_idx] != -100:
                        example_label.append(self.model_train_config.index2tag[label_ids[batch_idx, seq_idx]])
                        example_pred.append(self.model_train_config.index2tag[preds[batch_idx, seq_idx]])
                actual_label.append(example_label)
                pred_label.append(example_pred)
            
            return pred_label, actual_label
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    
    def compute_metrics(self, evaluate_prediction):
        try:
            y_pred, y_true = self.align_predictions(evaluate_prediction.predictions, evaluate_prediction.label_ids)
            return {"f1_score": f1_score(y_true, y_pred)}
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)
    def train(self):
        try:
            trainer = Trainer(model_init=self.model_init,
                              args=self.create_training_arguments(),
                              data_collator=self.data_collector(),
                              compute_metrics=self.compute_metrics,
                              train_dataset=self.preprocessed_data['train'].select(range(100)),
                              eval_dataset=self.preprocessed_data['validation'].select(range(100)),
                              tokenizer=self.model_train_config.tokenizer)
            
            self.my_logger.write_log('training started ......')
            result = trainer.train()
            self.my_logger.write_log(f" Result of the training {result} ")
            trainer.save_model(self.model_train_config.output_dir)
            self.my_logger.write_log(f"Trained Model saved at {self.model_train_config.output_dir}")
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())


            

if __name__ == '__main__':
    conf_ob = Configuration()
    data_ings = DataIngestion(conf_ob.get_data_ingestion_config())
    data_pre = DataPreprocessing(conf_ob.get_data_prepration_config(), data_ings.get_data())

    traing_config = conf_ob.get_model_training_config()

    ob_train = TrainTokenClassifier(traing_config, data_pre.prepare_data_for_fine_tuning())
    ob_train.train()


