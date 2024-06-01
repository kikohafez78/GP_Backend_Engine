from .model import BertModel
import torch.nn as nn
import torch
import numpy as np
import  torch.nn.functional as F 
# from ..crf_slot_filler.crf_torch import CRF
from TorchCRF import CRF
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class IntentClassifier(nn.Module):
    def __init__(self, input_dim: int, num_intent_labels: int, dropout_rate: float = 0.1):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)
        
    def forward(self, x: torch.Tensor):
        return self.linear(self.dropout(x)) 
    
    
    
class SlotClassifier(nn.Module):
    def __init__(self, input_dim: int, num_slot_lables: int, dropout_rate: float = 0.1):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_lables)
        
    def forward(self, x: torch.Tensor):
        return self.linear(self.dropout(x))



class JOINTIDSF:
    def __init__(self, config,model_name: str = "", intent_classes: list[str] = [], slot_tags: list[str] = None, num_layers: int = 1, layer_dims: list[int] = [], dropout: float = 0, window_mode: bool = False, crf_mode: bool = False):
        self.model_name = model_name
        self.classes = intent_classes
        self.tags = slot_tags
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.dropout = dropout
        self.window_mode = window_mode
        self.crf_mode = crf_mode
        self.window_size = None
        self.crf = None
        self.config = config    
        self.build()
        
    def build(self):
        if not self.window_mode and not self.crf_mode:
            self.bert = BertModel.from_pretrained(self.model_name)
            self.layers = nn.Sequential([self.bert])
            self.layers.append(nn.Linear(self.bert.config.hidden_size, self.layer_dims[0][0], True))
            for layer in range(0,self.num_layers):
                self.layers.append(nn.Linear(self.layer_dims[layer][0],self.layer_dims[layer][1], True))
                self.layers.append(nn.Tanh())
            self.layers.append(nn.Linear(self.layer_dims[self.num_layers][0], len(self.classes), True))
            self.layers.append(nn.Softmax(dim = 1))
            
        elif not self.window_mode and self.crf_mode:
            self.crf = CRF(len(self.tags))
            self.bert = BertModel.from_pretrained(self.model_name, return_dict = True)
            self.intent = IntentClassifier(self.bert.config.hidden_size, len(self.classes))
            self.slot_classifier = SlotClassifier(self.bert.config.hidden_size, len(self.tags))
        else:
            raise ValueError(" cannot be both at the same time") 
        self.initialize_linear_layers()
        
        
    def initialize_linear_layers(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    nn.init.zeros_(m.bias)
                    
        layer_prefix = "layer"  # You can customize the layer prefix here
        for i in range(1, self.num_layers):
            layer_name = f"{layer_prefix}{i+1}"
            # Check if the layer exists as an attribute
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
            if isinstance(layer, nn.Linear):
                layer.apply(init_weights)  # Apply initialization to the layer's weights and bias
            else:
                print(f"Warning: Layer '{layer_name}' not found in the model.")
                
    def model_metrics(self, lr: float = 1e-5, optimizer: str = None, loss_fn: str = None):
        self.lr = lr
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(lr = lr)
        elif optimizer.lower() == "sqd":
            self.optimizer = torch.optim.SGD(lr = lr)
        else:
            raise ValueError("optimizer not set correctly") 
        
        if loss_fn.lower() == "cross":
            self.loss_fn = F.cross_entropy
        elif loss_fn.lower() == "binary":
            self.loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise ValueError("loss function not set correctly")
        
    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_label_ids):
        if not self.window_mode and not self.crf_mode:
            return self.layers(input_ids)
        elif not self.window_mode and self.crf_mode:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            intent_logits = self.intent(pooled_output)
            slot_logits = self.slot_classifier(sequence_output)
            total_loss = 0
            if intent_label_ids is not None:
                if len(self.classes) == 1:
                    intent_loss_fct = nn.MSELoss()
                    intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1)) 
                else:
                    intent_loss_fct = nn.CrossEntropyLoss()
                    intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
                total_loss += intent_loss
            
            if slot_label_ids is not None:
                slot_loss = self.crf(slot_logits, slot_label_ids, mask = attention_mask.byte(), reduction = 'mean')
                slot_loss = -1 * slot_loss
                total_loss += self.config.slot_loss_coef * slot_loss
            outputs = ((intent_logits, slot_logits),)+ outputs[2:]
            outputs = (total_loss,) + outputs
            return outputs
        
    def training_step(self, batch):
        text, labels = batch 
        out = self(text)                  
        loss = self.loss_fn(out, labels) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = self.loss_fn(out, labels)   
        acc = accuracy_score(out, labels)           
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
     
    def evaluate(model, val_loader):
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)  
                   
    def train(self, train_data, val_data, epochs: int = 20, batch_size: int = 32):
        self.layers.train()
        if self.crf_mode:
            self.crf.train()
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_loader = DataLoader(val_data, batch_size, shuffle = True, num_workers=4, pin_memory = True)
        history = []
        optimizer = self.optimizer(self.layers.parameters(), self.lr)
        for epoch in tqdm(range(epochs)):
            # Training Phase 
            for batch in train_loader:
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = self.evaluate(self.layers, val_loader)
            self.epoch_end(epoch, result)
            history.append(result)
        return history
    
    
    def predict(self, sentence):
        return self(sentence)
        