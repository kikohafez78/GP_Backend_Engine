import nltk
import re
import pandas as pd
# from Module.action_module.actions.constants import constants

class masking:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config 
        
    def formula_parse_to_tree(self, formula: str):
        grammar = r"""
            S -> E
            E -> T | E OP T
            T -> F | T POW F
            F -> NUM | ID | LPAREN E RPAREN
            POW -> '^'
            OP -> '+' | '-' | '*' | '/'
            NUM -> '[0-9]+(\.[0-9]+)?'
            ID -> '[a-zA-Z][a-zA-Z0-9]*'
            LPAREN -> '('
            RPAREN -> ')'
        """

        cfg_parser = nltk.CFG.fromstring(grammar)
        parser = nltk.ChartParser(cfg_parser)
        # " ".join(formula.split())
        parsed_formula = parser.parse(nltk.word_tokenize(formula))
        if parsed_formula == None:
            return None
        else:
            parsed_string = " ".join(str(leaf) for leaf in parsed_formula[0].leaves())
            return parsed_string
        
    def condition_parse_to_tree(self, formula: str):
        grammar = r"""
            S -> E
            E -> C | E OP E
            C -> VAL CMP VAL | LPAREN E RPAREN
            VAL -> NUM | ID
            CMP -> '<' | '>' | '=' | '!=' | '<=' | '>=' | '==' 
            OP -> 'AND' | 'OR'
            NUM -> '[0-9]+(\.[0-9]+)?'
            ID -> '[a-zA-Z][a-zA-Z0-9]*'
            LPAREN -> '('
            RPAREN -> ')'
        """
        cfg_parser = nltk.CFG.fromstring(grammar)
        parser = nltk.ChartParser(cfg_parser)
        # " ".join(formula.split())
        parsed_formula = parser.parse(nltk.word_tokenize(formula))
        if parsed_formula == None:
            return None
        else:
            parsed_string = " ".join(str(leaf) for leaf in parsed_formula[0].leaves())
            return parsed_string
  
    def mask_corpus(self, corpus: str):
        pattern_Name = r"'(.*?)'"
        pattern_Range = r"""\b[a-zA-Z]+[0-9]+:[a-zA-Z]+[0-9]+\b"""
        pattern_cell_ref =r"""\b[a-zA-Z]+[0-9]+\b"""
        pattern_number = r"""\b[0-9]+(\.[0-9]+)?\b"""
        pattern_punctuation = r"""[\?!\.]"""
        # pattern_formula = r"""([a-zA-Z]*[0-9] *[\+\-\/\*])"""
        # formula = self.formula_parse_to_tree(corpus)
        # condition = self.condition_parse_to_tree(corpus)
        text = re.sub(pattern_Name, '<Name>', corpus)
        # print("after sheet name: ", text, re.findall(pattern_Name, corpus))
        text = re.sub(pattern_Range, '<Range>', text)
        # print("after range object: ", text)
        text = re.sub(pattern_cell_ref, '<Cell>', text)
        # print("after cell reference: ", text)
        text = re.sub(pattern_number, '<Number>', text)
        # print("after number: ", text)
        text = re.sub(pattern_punctuation, '', text)
        # print("after punctuation: ", text)
        # if formula is not None:
        #     text.replace(formula , '<Formula>')
        # print("after formula: ", text)
        # if condition is not None:
        #     text.replace(condition, '<Condition>')
        # print("after condition: ", text)
        # chart_indexes = [i for i, w in enumerate(text.lower().split()) if w == 'chart']
        # table_indexes = [i for i, w in enumerate(text.lower().split()) if w == 'table']
        # print("chart objects indexes: ",chart_indexes)
        # print("pivot table object indexes: ",table_indexes)
        return text
    def mask_corpus_(self, corpus: pd.Series):
        i = 0
        for text in corpus:
            if i < 20:
                print("before: ", text)
            text = self.mask_corpus(text)
            if i < 20:
                print("after: ", text)
            i += 1
        return corpus
# masker = masking("masker",{})
# print(masker.mask_corpus("create a new sheet called 'hero'  and insert a new column called 'arnold'"))