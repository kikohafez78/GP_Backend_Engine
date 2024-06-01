from Module.preprocessing.preprocessing import preprocessing
from Module.feature_extraction.ExcelToJson import excel_information_extractor
from Module.feature_extraction.text_feature_extraction import text_features
from Module.action_module.feature_resolution import feature_resolution
from Module.action_module.feature_resolution_config import get_feature_resolution
# from Module.coreference.main import coreference_resolution_seive
#==========================================================================
# from Module.self_instruct.self_Instruct_config import get_self_instruct
# from Module.self_instruct.self_instruct import self_instruct
#==========================================================================
from Module.intention.classical.Integrated_ISF import classical_Integrated
from Module.intention.deep.bert_model_implementation_torch.intention_model import JOINTIDSF
#==========================================================================
from Auto_Config import get_auto_config
from Module.intention.deep import deep_model_config
#==========================================================================
from Module.action_module.actions.entryandmanipulation import entry_manipulation_App
from Module.action_module.actions.charts import Charts_App
from Module.action_module.actions.formatting import formatting_App
from Module.action_module.actions.management import management_App
from Module.action_module.actions.pivot_table import Pivot_App
from Module.action_module.actions.formula import Formula_App
#==========================================================================
import os
import sys
import time
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
#==========================================================================
import glob2 as gl 
import speech_recognition as sr
import pyttsx3
import xlwings as xlsx


def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
    
def listen():
    MySpeech = None
    r = sr.Recognizer() 
    while(1):    
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                SpeakText(MyText)
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")
    MySpeech = MyText if MyText != None else MySpeech
    return MySpeech




class Auto:
    config = get_auto_config()
    speech_recognition = sr.Recognizer()
    preprocessing = preprocessing()
    sheet_information_extraction = excel_information_extractor(False, "excel dissector")
    prompt_features = text_features
    coreference_resolution = None
    sentence_simplification = None
    #=======================
    simplifier_module = None
    classical_planning_module = classical_Integrated("planning")
    deep_planning_module = None
    feature_resolver = feature_resolution(get_feature_resolution())
    #========================
    # self_instruction = self_instruct("instructor", get_self_instruct())
    #========================
    entry = entry_manipulation_App()
    management = management_App()
    formatting = formatting_App()
    charts = Charts_App()
    pivot = Pivot_App()
    formula = Formula_App()
    #========================
    working_sheet = None
    def __init__(self, config: dict):
        self.config = config
        self.state = {}
        self.execution_state = {}
        
    def set_current_sheet(self, user_name: str, filename: str):
        self.working_sheet = self.feature_extraction.file_scan_(os.path.join(os.getcwd(), f"\\{user_name}\\{filename}"))
    
    def extract_features_from_prompt(self, prompt: str):
        return self.text_features()
    
    def text_preprocessing(self, text):
        text_state = self.preprocessing(text)
        return text_state
        
    def get_working_sheets_(self, sheet_names):
        pass
    
    def coreference(self, text):
        pass
    
    def speech_recognition_(self):
        return listen()
    
    def planning_(self, json):
        return
    
    def action_mapping(self, actions):
        pass
    
    def execution_(self, json):
        return
    
    def error_checking_(self, state):
        return
    
    # def self_instruct_(self, text, mode, intents = None):
    #     if mode == "prompt":
    #         return self.self_instruction.generate_parphrases_based_on_s(text)
    #     elif mode == "plan":
    #         return self.self_instruction.augment_auto(text, intents)
    #     elif mode == "learn":
    #         data = self.self_instruction.augment_auto(text, intents)
            
        
    def demo_test_1(self,path):
        steps = []
        self.entry.OpenWorkbook(path)
        time.sleep(3)
        steps.append(self.entry.autofill(path, "Sheet1", "D2", "Sheet1", "D2:D9", path))
        self.entry.Save()
        self.entry.SaveWorkbook(path)
        # self.entry.closeWorkBook()
        return steps, []
    
    def demo_test_2(self,path):
        steps = []
        time.sleep(3)
        steps.append(self.formatting.conditional_formatting(path, "Sheet1", "D2:D9", fillColor="yellow", formula="=D2>150000"))
        self.entry.Save()
        self.entry.SaveWorkbook(path)
        # self.entry.closeWorkBook()
        return steps, []
    
    def demo_test_3(self,path):
        steps = []
        time.sleep(3)
        steps.append(self.charts.CreateChart(path, source="A1:D9", destSheet="Sheet1", chartName="AboIsmail", chartType="Pie", XField=1, YField=[4]))
        self.entry.Save()
        self.entry.SaveWorkbook(path)
        # self.entry.closeWorkBook()
        return steps, []

# config = get_auto_config()
# auto = Auto(config)
# auto.demo_test_1(os.path.join(os.getcwd(), "./IncomeStatement.xlsx"))
        
        
