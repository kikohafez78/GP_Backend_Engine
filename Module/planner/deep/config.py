from pathlib import Path

def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 10**-8,
        "seq_len": 512,
        "d_model": 768,
        "num_layers": 12,
        "d_ff": 3072,
        "heads": 12,
        "datasource": 'dataset',
        "lang_src": "instruction input",
        "lang_tgt": "output",
        "model_folder": "weights",
        "model_basename": "planner",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
