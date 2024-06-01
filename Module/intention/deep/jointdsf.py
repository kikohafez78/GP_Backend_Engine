from full_model import JointBert
from intent_model import intent_model
from isf_datasets import full_dataset
from transformers import  get_linear_schedule_with_warmup, EarlyStoppingCallback, BertTokenizer, AdamW, Trainer, TrainingArguments, get_scheduler
from deep_model_config import get_deep_model_config
import numpy as np
import torch
import pandas as pd
from bert_utils import replace_angle_brackets, mask_tokens
import sys 
import os
import torch.nn.functional as F
from tqdm import tqdm
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class model:
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained(config["model_dir"])
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None
        self.intent_labels, self.slot_labels = None, None
        if config["mode"] == "intention":
            self.model = intent_model(config, [], config["dropout_rate"])
        else:
            self.model = JointBert(config, [], [])
        
    def set_preprocessor_func(self, func):
        self.preprocessor = func
    
    def train(self):
        dataset = full_dataset(self.config, self.tokenizer)
        self.intent_labels, self.slot_labels = dataset.intent_classes, dataset.slots_classes
        train_loader, val_loader = dataset.get_loaders()
        if self.config["mode"] == "intention":
            optimizer = torch.optim.Adam(self.model.parameters(), self.config["lr"], weight_decay = self.config["weight_decay"])
            self.train_intention(self.model, self.config["epochs"], train_loader, val_loader, optimizer, self.config["ckpt_path"], self.config["best_model_path"])
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.config["lr"])
            total_steps = len(train_loader) * self.config["epochs"]
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            self.train_isf(self.model, train_loader, val_loader, optimizer, scheduler, self.config["epochs"], self.device)
        
        
    def predict(self, text: str):
        text = self.preprocessor(text)
        if self.config["mode"] == "intention":
            self.predict_intent(text)
        else:
            self.predict_isf(text)
            
            
    def set_loss_fn(self, func):
        self.loss_fn = func

    def load_ckpt(self, ckpt_path, model, optimizer):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['state_dict'])
        valid_loss_min = ckpt['valid_loss_min']
        return model, optimizer, ckpt['epoch'], valid_loss_min.item()
    
    def save_ckpt(self, state, is_best, ckpt_path, best_model_path):
        f_path= ckpt_path
        torch.save(state, f_path)
        if is_best:
            best_f_path = best_model_path
            shutil.copyfile(f_path, best_f_path)
    

    def train_intention(self, model, epochs, train_loader, val_loader, optimizer, ckpt_path, best_model_path):
        for epoch in range(1, epochs + 1):
            train_loss = 0
            val_loss = 0
            model.train()
            for batch_index, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = batch["attention_mask"].to(self.device, dtype=torch.long)
                token_type_ids = batch["token_type_ids"].to(self.device, dtype=torch.long)
                targets = batch["targets"].to(self.device, dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = model(input_ids, token_type_ids, attention_mask)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += (1 / (batch_index + 1)) * (loss.item() - train_loss)
            print(f"Epoch {epoch} ended with train loss of {train_loss}")
            
            model.eval()
            with torch.no_grad():
                for batch_index, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input_ids = batch['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = batch["attention_mask"].to(self.device, dtype=torch.long)
                    token_type_ids = batch["token_type_ids"].to(self.device, dtype=torch.long)
                    targets = batch["targets"].to(self.device, dtype=torch.long)
                    
                    outputs = model(input_ids, token_type_ids, attention_mask)
                    loss = self.loss_fn(outputs, targets)
                    
                    val_loss += (1 / (batch_index + 1)) * (loss.item() - val_loss)
            print(f"Epoch {epoch} ended with val loss of {val_loss}")
            
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            self.save_ckpt(checkpoint, False, ckpt_path, best_model_path) 

        return model
    
    
    def updated_intent_training(self, num_epochs, train_dataset, eval_dataset, model, lr):
        training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        )
        optimizer = AdamW(model.parameters(), lr=lr)
        num_training_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        early_stopping_patience = 3
        early_stopping = EarlyStoppingCallback(early_stopping_patience)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[early_stopping]
        )
        trainer.train()
        return model

    
    def train_isf(self, model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
        model.to(device)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_targets'].to(device)
                slot_labels = batch['slot_targets'].to(device)

                optimizer.zero_grad()

                slot_loss, intent_logits = model(input_ids, token_type_ids, attention_mask, slot_labels)
                intent_loss = self.loss_fn(intent_logits, intent_labels)
                loss = slot_loss + intent_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Training loss: {avg_train_loss}")

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    intent_labels = batch['intent_targets'].to(device)
                    slot_labels = batch['slot_targets'].to(device)

                    slot_loss, intent_logits = model(input_ids, token_type_ids, attention_mask, slot_labels)
                    intent_loss = self.loss_fn(intent_logits, intent_labels)
                    loss = slot_loss + intent_loss

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Validation loss: {avg_val_loss}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print("Saved the best model!")
                return model
                
    def predict_intent(self, text: str):
        inputs = self.tokenizer(            
            text,
            None,
            add_special_tokens = True,
            return_attention_mask = True,
            return_tensors = 'pt',
            return_token_type_ids = True,
            padding = 'max_length',
            max_length = self.config["max_len"],
            truncation = True
            )
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        device = self.device
        self.model.to(device)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, token_type_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        predicted_label_idx = torch.argmax(probs, dim=1).item()
        print(predicted_label_idx,torch.max(probs, dim=1))
        predicted_label = self.intent_labels[predicted_label_idx]
        return predicted_label, probs[0].cpu().numpy()


    
    def predict_isf(self, prompt: str):
        self.model.eval()
        inputs = self.tokenizer.encode_plus(
            prompt,
            None,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=True,
            padding='max_length',
            max_length=self.conifg["max_len"],
            truncation=True
        )
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            slot_predictions, intent_logits = model(input_ids, token_type_ids, attention_mask)
        intent_pred = torch.argmax(intent_logits, dim=1).item()
        intent_label = self.intent_labels[intent_pred]
        slot_predictions = slot_predictions[0]
        slot_labels_pred = [self.slot_labels[slot] for slot in slot_predictions]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = [token for token, mask in zip(tokens, attention_mask[0]) if mask == 1]
        slot_labels_pred = slot_labels_pred[:len(tokens)]
        return intent_label, list(zip(tokens, slot_labels_pred))

    
