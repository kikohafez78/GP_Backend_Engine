import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def get_deep_model_config():
    current_dir = os.getcwd()
    return {
        'max_len': 256,
        'batch_size': 8,
        'epochs': 10,
        'lr':1e-05,
        'out_first_layer': 768,
        'dropout_rate': 0.3,
        'model_dir':'bert-base-uncased',
        'ckpt_path': os.path.join(current_dir, "Module\\intention\\deep\\ckpts"),
        'ckpt_model_path': os.path.join(current_dir, "Module\\intention\\deep\\experiments"),
        'data_file': os.path.join(current_dir, "Module\\intention\\data\\Book1.csv"),
        'slots_file': os.path.join(os.getcwd(), f"Module\\intent\\deep\\slots.json"),
        'mode': "intention",
        'optimizer': "adam",
        'early_stopping': True,
        'train_fraction': 0.9,
        'weight_decay': 0.01
    }