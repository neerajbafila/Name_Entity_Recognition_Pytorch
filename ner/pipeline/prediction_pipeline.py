from ner.exception_and_logger.logger import logger
from ner.configration.configurations import Configuration
from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.constants import *
import sys
import torch
import numpy as np

class PredictionPipeline:
    def __init__(self, config: Configuration):
        self.config = config
        self.my_logger = logger(CONFIG_FILE_NAME)
    
    def prediction_pipeline(self, data:str):
        try:
            data = data.split()
            prediction_pipeline_config = self.config.get_prediction_config()
            tokenizer = prediction_pipeline_config.tokenizer
            output_dir = prediction_pipeline_config.output_dir
            truncation = prediction_pipeline_config.truncation
            is_split_into_words = prediction_pipeline_config.is_split_into_words
            index2tag = prediction_pipeline_config.index2tag
            tag2index = prediction_pipeline_config.tag2index
            fine_tuned_model = prediction_pipeline_config.fine_tuned_model
            input_ids = tokenizer(data, truncation=truncation, is_split_into_words=is_split_into_words, return_tensors="pt")['input_ids']
            # formatted_data = torch.tensor(input_ids['input_ids']).reshape(-1,1)
            # print(input_ids)
            # print(fine_tuned_model)
            model = XLMRobertaForTokenClassification.from_pretrained(fine_tuned_model)
            # print(model.config)
            outputs = model(input_ids).logits
            predictions = np.argmax(outputs.detach().numpy(), axis=-1)
            # print(predictions)
            pred_tags = [index2tag[idx] for idx in predictions[0][1:-1]]
            return pred_tags

        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys.exc_info())
    
    def run_prediction_pipeline(self, data):
        predictions = self.prediction_pipeline(data)
        response = {"Input_data": data,
                    "prediction": predictions
        }
        print(response)
        return response


if __name__ == "__main__":

    ob_config = Configuration()
    ob = PredictionPipeline(ob_config)
    ex = "List of years in Brazil. My name is Neeraj Bafila"
    ob.run_prediction_pipeline(ex)