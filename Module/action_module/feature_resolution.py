from actions.charts import Charts_App
from actions.entryandmanipulation import enntry_manipulation_App
from actions.formatting import formatting_App
from actions.management import management_App
from actions.pivot_table import Pivot_App
from actions.formula import Formula_App



class feature_resolution:
    charts_app = Charts_App()
    entry_app = enntry_manipulation_App()
    formatting_app = formatting_App()
    management_app = management_App()
    pivot_app = Pivot_App()
    formula_app = Formula_App()
    
    def __init__(self, config):
        self.config = config
        self.current_state = None
    
    def update_state(self, current_json):
        self.current_state = current_json
    
    def resolve(self, class_, intent_,  text, features_):
        steps = []
        if class_ == "entry and manipulation":
            self.entry_n_man(intent_, features_)
        elif class_ == "management":
            self.management(intent_, features_)
        elif class_ == "formatting":
            self.formatting(intent_, features_)
        elif class_ == "charts":
            self.charts(intent_, features_)
        elif class_ == "pivot table":
            self.pivot(intent_, features_)
        else:
            self.formula(intent_, features_)
    