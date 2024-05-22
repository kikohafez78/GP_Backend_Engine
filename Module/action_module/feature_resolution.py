import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


class feature_resolution:
    
    def __init__(self, config):
        self.config = config
        self.current_state = None
    
    def update_state(self, current_json):
        self.current_state = current_json
    
    def resolve(self, class_, intent_, features_, workbook_definition_json: dict):
        step = None
        if class_ == "entry and manipulation":
            step = self.entry_n_man(intent_, features_)
        elif class_ == "management":
            step = self.management(intent_, features_)
        elif class_ == "formatting":
            step = self.formatting(intent_, features_)
        elif class_ == "charts":
            step = self.charts(intent_, features_)
        elif class_ == "pivot table":
            step = self.pivot(intent_, features_)
        else:
            step = self.formula(intent_, features_)
    
    def entry_n_man(self, intent: str, features: list[tuple]):
        pass
    
    def management(self, intent: str, features: list[tuple]):
        pass
    
    def formatting(self, intent: str, features: list[tuple]):
        pass
    
    def charts(self, intent: str, features: list[tuple]):
        pass
    
    def pivot(self, intent: str, features: list[tuple]):
        pass
    
    def formula(self, intent: str, features: list[tuple]):
        pass