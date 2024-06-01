from transformers import BertTokenizer
import torch 
import numpy as np
import pandas as pd
import sys
import os
from torch.utils.data import DataLoader
from bert_utils import replace_angle_brackets
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class intent_dataset:
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.classes = df['intent'].unique().tolist()
        self.y = df['intent']
        self.x = df['prompt']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        title = str(self.x[index])
        title = ''.join(title.split())
        title = replace_angle_brackets(title)  # Assuming mask_tokens is a function defined elsewhere
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=True,
            padding='max_length',
            max_length=self.max_len,
            truncation=True
        )
        target = self.classes.index(self.y[index])  # Get the class index
        return {
            'input_ids': inputs["input_ids"].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)  # Directly use class index
        }
        
        
class isf_dataset:
    def __init__(self, df: pd.DataFrame, intent_classes: list[str], slot_classes: list[str], tokenizer: BertTokenizer, max_len: int):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.intent_classes = intent_classes
        self.slot_classes = slot_classes
        self.intents = df['intent']
        self.slots = df["slots"]
        self.x = df['prompt']
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        title = str(self.x[index])
        title= ''.join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens = True,
            return_attention_mask = True,
            return_tensors = 'pt',
            return_token_type_ids = True,
            padding = 'max_length',
            max_length = self.max_len,
            truncation = True
        )
        intent_target = self.intent_classes.index(self.intents[index])
        slot_targets = [self.slot_classes.index(slot) for slot in list(self.slots[index])][:self.max_len + 1]
        return {
            'input_ids': inputs["input_ids"].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'intent_targets': torch.tensor(intent_target, dtype=torch.long),
            'slot_targets': torch.tensor(slot_targets, dtype=torch.long)
        }
        

class full_dataset:
    def __init__(self, config: dict, tokenizer: BertTokenizer):
        self.config = config
        self.df = pd.read_csv(os.path.join(os.getcwd(),f"intent\\data\\{config['data_file']}"))
        self.train = self.df.sample(frac=config["train_fraction"], random_state=200).reset_index(drop=True)
        self.val = self.df.drop(self.train.index).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.intent_classes = self.df['intent'].unique().tolist()
        if config["mode"] == "intention":
            self.train = intent_dataset(self.train, tokenizer, config["max_len"])
            self.val = intent_dataset(self.val, tokenizer, config["max_len"])
        else:
            self.slots_classes: dict = json.loads(open(config['slots_file'], "r"))
            self.train = isf_dataset(self.train, self.intent_classes, self.slots_classes, self.tokenizer, config["max_len"])
            self.val = isf_dataset(self.val, self.intent_classes, self.slots_classes, self.tokenizer, config["max_len"])
        
        self.train = DataLoader(
            self.train,
            shuffle = True,
            batch_size = config["batch_size"],
            num_workers = 0
            )

        self.val = DataLoader(
            self.val,
            shuffle = False,
            batch_size = config["batch_size"],
            num_workers = 0
            )
        
    def get_loaders(self):
        return self.train, self.val
    
    def get_tags(self):
        return self.intent_classes ,self.slots_classes
    
