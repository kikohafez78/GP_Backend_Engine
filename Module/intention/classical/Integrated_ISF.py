import pandas as pd
from classical_config import get_classical_config
from integrated_model import integrated_model_v_4
from CRFModel import CRFModel
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class classical_Integrated:
    
    intent_model = integrated_model_v_4(get_classical_config())
    slot_filling = CRFModel(None)
    
    def __init__(self, model_name):
        self.name = model_name
        self.intent_model.load_model()
        self.slot_filling.load_model("CRF")
        
    def predict(self, X):
        predictions  = []
        for x in X:
            _, Intent, _ = self.intent_model.predict([x])
            filled_slots = self.slot_filling.predict([(x, Intent)])
            predictions.append((Intent, filled_slots))
        return predictions
            
            
model = classical_Integrated("ISF")
print(model.predict(["create a new sheet called 'hero'"]))