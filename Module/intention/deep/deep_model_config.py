import os
def get_deep_model_config():
    current_dir = os.getcwd()
    return {
        'max_len': 256,
        'batch_size': 8,
        'epochs': 10,
        'lr':1e-05,
        'out_first_layer': 768,
        'dropout_rate': 0.1,
        'model_dir':'bert-base-uncased',
        'ckpt_path': os.path.join(current_dir, "Module\\intention\\deep\\ckpts"),
        'ckpt_model_path': os.path.join(current_dir, "Module\\intention\\deep\\experiments"),
        
    }