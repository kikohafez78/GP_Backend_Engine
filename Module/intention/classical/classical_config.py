import os 
def get_classical_config():
    current_dir = os.getcwd()
    return {
        "similarity_factor": "cosine",
        "max_iter": 32,
        "num_classes": 6,
        "classes": [],
        "num_sub_classes": 93,
        "sub_classes": [],
        "padding_token": "[PAD]",
        "padding_token_id": 0,
        "base_class_column": "classes",
        "sub_class_column": "intent",
        "input_column": "prompt",
        "alpha": 1,
        "force_alpha": "warn",
        "fit_prior": True,
        "multi_class": "ovr",
        "data_path": os.path.join(current_dir, "Module\\intention\\data\\Book1.csv"),
        "model_dir": os.path.join(current_dir, "Module\\intention\\classical\\IntentionModelFiles"),
        "misc_dir": os.path.join(current_dir,"Module\\intention\\classical\\"),
        "max_seq_len": 300,
        "do_eval": True,
        "base_model_name": "base",
        "catboost_parameters":{
            'iterations': 300,
            'learning_rate': 0.01,
            'eval_metric': 'Accuracy',
            'task_type': 'GPU',
            'early_stopping_rounds': 20,
            'use_best_model': True,
            'verbose': 50
        },
        "cat_features": ["classes"],
        "text_features": ["prompt"],
        "sub_model": os.path.join(current_dir, "Module\\intention\\classical\\IntentionModelFiles\\tree_model")
    }