import torch
import torch.nn as nn
from transformers import BertModel
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class intent_model(nn.Module):
    def __init__(self, config: dict, intent_labels: list[str], dropout: float = 0.1):
        super(intent_model,self).__init__()
        self.config = config
        self.intent_labels = intent_labels
        self.num_intents = len(intent_labels)
        self.dropout_rate = dropout
        self.bert = BertModel.from_pretrained(self.config["model_dir"])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_1 = nn.Linear(self.bert.config.hidden_size,self.config["out_first_layer"])
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(self.config["out_first_layer"], len(self.intent_labels))
        
    def set_intent_labels(self,intent_labels):
        self.intent_labels = intent_labels
        self.num_intents = len(intent_labels)
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        output_dropout = self.dropout(output.pooler_output)
        out_layer_1 = self.layer_1(output_dropout)
        act_1 = self.activation_1(out_layer_1)
        out_layer_2 = self.layer_2(act_1)
        return out_layer_2
        