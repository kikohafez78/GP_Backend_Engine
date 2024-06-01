from transformers import BertModel
import torch
import torch.nn as nn
import TorchCRF
import os
import sys 
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


class JointBert(nn.Module):
    
    def __init__(self, config: dict, intent_labels: list[str], slot_labels: list[str]):
        self.config = config
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels
        self.num_intents = len(intent_labels)
        self.num_slots = len(slot_labels)
        self.bert = BertModel.from_pretrained(self.config["model_dir"])
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.layer_1 = nn.Linear(self.bert.config.hidden_size,self.config["out_first_layer"])
        self.intent_activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(self.config["out_first_layer"], len(self.intent_labels))
        self.crf = TorchCRF.CRF(self.num_slots, True)
        self.slot_layer_1 = nn.Linear(self.bert.config.hidden_size, self.config["out_first_layer"])
        self.slot_activation_1 = nn.ReLU()
        self.slot_layer_2 = nn.Linear(self.config["out_first_layer"], len(self.slot_labels))
        
    def set_intent_labels(self, intent_labels):
        self.intent_labels = intent_labels
        self.num_intents = len(intent_labels)
        
        
    def set_slot_labels(self, slot_labels):
        self.slot_labels = slot_labels
        self.num_slots = len(slot_labels)    
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = output.pooler_output
        sequence_output = output.last_hidden_state
        
        output_dropout = self.dropout(pooled_output)
        intent_output_layer_1 = self.layer_1(output_dropout)
        intent_activation_1 = self.intent_activation_1(intent_output_layer_1)
        intent_prediction = self.layer_2(intent_activation_1)
        
        slot_output_dropout = self.dropout(sequence_output)
        slot_out_layer_1 = self.slot_layer_1(slot_output_dropout)
        slot_activation_1 = self.slot_activation_1(slot_out_layer_1)
        slot_predictions = self.slot_layer_2(slot_activation_1)
        
        if labels is not None:
            loss = -self.crf(slot_predictions, labels, mask = attention_mask.bool(), reduction = 'mean')
            return loss, intent_prediction
        else:
            slot_predictions = self.crf.decode(slot_predictions, mask=attention_mask.bool())
            return intent_prediction, slot_predictions
        
        

