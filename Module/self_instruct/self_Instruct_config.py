import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def get_self_instruct():
    return {
        "n_predictions": 10,
        "mode": "paragraph"
    }