from ner.exception_and_logger.logger import logger
from ner.constants import *
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import XLMRobertaConfig
import sys
class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    def __init__(self, config:dict):
        """Config : Contains configuration for XLM roberta model type dict
        needs to be passed from user after configuration over-ridding.
        """
        self.my_logger = logger(CONFIG_FILE_NAME)
        super().__init__(config)
        self.my_logger.write_log("Model inited")
        self.num_labels = config.num_labels
        # # Load model body
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout_layer = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None, **kwargs):
    # def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        try:
             # Use model body to get encoder representations
            outputs = self.roberta(input_ids, attention_mask, token_type_ids=token_type_ids, **kwargs)
            # Apply classifier to encoder representation
            sequence_outputs = self.dropout_layer(outputs[0])
            logits = self.linear(sequence_outputs)
            # calculate loss
            loss = None
            if labels is not None:
                loss_fnt = nn.CrossEntropyLoss()    
                loss = loss_fnt(logits.view(-1, self.num_labels), labels.view(-1))
            # Return model output object

            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                         attentions=outputs.attentions)
        except Exception as e:
            self.my_logger.write_exception(e)
            raise Exception(e, sys)

# import torch
# from transformers import AutoConfig, AutoTokenizer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# xlmr_model_name = "xlm-roberta-base"
# cache_dir_for_dnld = 'D:\\iNeuron\\FullStack\\DeepLearning\\NLP\\Name_Entity_Recognition_Pytorch\\data'
# xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name, cache_dir=cache_dir_for_dnld)
# tags= ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
# idx_tag = {idx: tag for idx, tag in enumerate(tags) }
# tag_idx = {tag: idx for idx, tag in enumerate(tags)}
# xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, num_labels=7, cache_dir=cache_dir_for_dnld, id2label=idx_tag, label2id=tag_idx)
# ob = XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config, cache_dir=cache_dir_for_dnld).to(device)
# my_inp = "This is testing"
# input_ids = xlmr_tokenizer(my_inp, return_tensors='pt')['input_ids']

# opt = ob(input_ids.to(device)).logits
# print(opt)
# pred = torch.argmax(opt, dim=-1)
# print(pred)




