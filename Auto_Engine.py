from Module.preprocessing.preprocessing import preprocessing
from Module.feature_extraction.ExcelToJson import excel_information_extractor
from Module.feature_extraction.text_feature_extraction import text_features
# from Module.coreference.main import coreference_resolution_seive
#==========================================================================
from Module.action_module.actions.charts import Charts_App
from Module.action_module.actions.entryandmanipulation import enntry_manipulation_App
from Module.action_module.actions.formatting import formatting_App
from Module.action_module.actions.management import management_App
from Module.action_module.actions.pivot_table import Pivot_App
from Module.action_module.actions.formula import formula_App
#==========================================================================
from Module.self_instruct.self_Instruct_config import get_self_instruct
from Module.self_instruct.self_instruct import self_instruct
#==========================================================================
from Module.intention.classical.Integrated_ISF import classical_Integrated
from Module.intention.deep.bert_model_implementation_torch.intention_model import JOINTIDSF
#==========================================================================
from Auto_Config import get_auto_config

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




class Auto_Engine:
    config = get_auto_config()
    speech_recognition = sr.Recognizer()
    preprocessing = preprocessing()
    feature_extraction = excel_information_extractor(False, "excel dissector")
    text_features = text_features
    coreference_resolution = coreference_resolution_seive
    sentence_simplification = None
    #=======================
    simplifier_module = None
    classical_planning_module = classical_Integrated("planning")
    deep_planning_module = None
    feature_resolver = None
    #=======================
    #===== actions =========
    charts_action = Charts_App()
    pivot_action = Pivot_App()
    manipulation_action = enntry_manipulation_App()
    management_action = management_App()
    formatting_action = formatting_App()
    formula_action = None
    #========================
    self_instruction = self_instruct("instructor", get_self_instruct())
    #========================
    working_sheets = {}
    def __init__(self, config: dict):
        self.config = config
        self.state = {}
        self.execution_state = {}
        
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
    
    def self_instruct_(self, text, mode, intents = None):
        if mode == "prompt":
            return self.self_instruction.generate_parphrases_based_on_s(text)
        elif mode == "plan":
            return self.self_instruction.augment_auto(text, intents)
        elif mode == "learn":
            data = self.self_instruction.augment_auto(text, intents)
            
        
    def run(self):
        pass        
    
    

        
        
