import re
from .normalization import simplify_punctuation, normalize_contractions,  spell_correction, Normalizer
from tokenization import Tokenizer, Sentencizer
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import json

currency_file = os.path.join(os.path.dirname(__file__),"./preloaded/dictionaries/currencies/currency.json")

def key_swapper(dictionary: dict):
    swapped_dict = {value: key for key, value in dictionary.items()}
    return swapped_dict

class preprocessing:
    def __init__(self):
        return
    
    @staticmethod
    def extract_urls(paragraph: str):
        paragraph = paragraph.lower()
        pattern = r"""\bhttps?://[^\s]+|\bwww\.[^\s]+"""
        urls_ids = []
        def replacer(match):
            urls_ids.append(match)
            return "<Url>"
        result = re.sub(pattern, replacer, paragraph)
        return result, urls_ids
    
    @staticmethod
    def extract_hashtags(paragraph: str):
        paragraph = paragraph.lower()
        hashtag = r"\b#\w+\b"
        hashtags_ids = []
        def replacer(match):
            hashtags_ids.append(match)
            return "<Hashtag>"
        result = re.sub(hashtag, replacer, paragraph)
        return result, hashtags_ids
    
    @staticmethod
    def special_characters(paragraph: str):
        paragraph = paragraph.lower()
        paragraph.replace("&"," and ")
        paragraph.replace("|"," or ")
        paragraph.replace("~"," not ")
        paragraph.replace("%"," percent ")
        paragraph.replace("@","")
        paragraph.replace("!",".")
        paragraph.replace("?",".")
        paragraph.replace("+ve", "positive")
        paragraph.replace("-ve", "negative")
        # paragraph.replace(">=", "more than or equal to")
        # paragraph.replace("<=", "less than or equal to")
        # paragraph.replace("<","less than")
        # paragraph.replace(">", "more than")
        # paragraph.replace("==", "equal to")
        # paragraph.replace("=", "equal to")
        currencies: dict = json.load(open(currency_file))
        for currency in currencies.keys():
            paragraph.replace(" " + currency + " ", currencies[currency])
        return paragraph
    
    @staticmethod
    def extract_formulas(paragraph: str):
        formulas = r"""
                (?:    
                    [+\-*/^<>≤≥∞] |  
                    [₀-₉] |  
                    [α-ωΑ-ΩΓΠΣΔΦΛΞΨΩ] |  
                    ∈∉∂∇ℏ∰⤼…    
                ) |  
                (?:    
                    \w+(?:\(.+\))?    
                ) |  
                (?:    
                    (?<![a-zA-Z0-9_])   
                    (?:_[₀-₉α-ωΑ-ΩΓΠΣΔΦΛΞΨΩ] |    
                        [^a-zA-Z0-9_]\^    
                    )
                    (?=[^\s\.,;:!?])   
                ) |  
                (?:    
                    [A-Za-z0-9_]+[']?   
                ) |  
                (?:    
                    \frac\{.+?\}\{.+?} |
                    \dfrac\{.+?\}\{.+?} |
                    \(.+\)/\(.+\)         
                ) |  
                (?:    
                    \frac{d}{dx}\(.+\) |   
                    \int\limits?_{.+}^{.+}(.+?)dx   
                ) |  
                (?:    
                    \[.+?\] |          
                    \det\(.+\)           
                ) |  
                (?:    
                    (?:=|≠|≤|≥|<|>|∈|∉)  
                    \(.+\)\s*(?:[=≠≤≥|<|>|∈|∉])\s*\(.+\)   
                )
                """
        basic_formulas = r"""\b[\w]+[\s]*[\+\-\*\^\%/][\s]*[\w]+[\s]*=[\s]*[\w]+\b"""
        functional_formulas = r"""\b[(normal)|(std)|(mode)|(mean)|(sum)|(log)|(ln)|(log)|(sin)|(cos)|(tan)|(sec)|(csc)|(cot)]\([\s]*[\w]+[\s]*\)[\s]*=[\s]*[\w]+\b"""
        complex_formulas = r"""\b[(normal)|(std)|(mode)|(mean)|(sum)|(log)|(ln)|(log)|(sin)|(cos)|(tan)|(sec)|(csc)|(cot)][0-9]*\([\s]*[\w]+[\s]*\)[\s]*[\+\-\*\^\%/][\s]*[(normal)|(std)|(mode)|(mean)|(sum)|(log)|(ln)|(log)|(sin)|(cos)|(tan)|(sec)|(csc)|(cot)][0-9]*\([\s]*[\w+][\s]*\)[\s]*=[\s]*[\w]+\b"""
        paragraph = paragraph.lower()
        formulas_ids = []
        def replacer(match):
            formulas_ids.append(match)
            return "<Formula>"
        intermidiate_result1 = re.sub(basic_formulas, replacer, paragraph)
        intermidiate_result2 = re.sub(functional_formulas, replacer, intermidiate_result1)
        end_result = re.sub(complex_formulas, replacer, intermidiate_result2)
        return end_result, formulas_ids
    
    @staticmethod
    def extract_dates(paragraph: str):
        paragraph = paragraph.lower()
        result_in_alphabet = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
        result_in_numerals = r"(?<!\d)\d{1,2}/\d{1,2}/\d{4}(?!\d)"
        dates_ids = []
        def replacer(match):
            dates_ids.append(match)
            return "<Date>"
        inter_result = re.sub(result_in_alphabet, replacer, paragraph)
        end_result = re.sub(result_in_numerals, replacer, inter_result)
        return end_result, dates_ids
    
    @staticmethod
    def extract_emails(pagagraph: str):
        paragrph = pagagraph.lower()
        email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        emails_ids = []
        def replacer(match):
            emails_ids.append(match)
            return "<Email>"
        result = re.sub(email_regex, replacer, paragrph)
        return result, emails_ids
    
    @staticmethod
    def extract_excel_entites(paragraph: str):
        range_regex = r"""\b[a-zA-Z]+[0-9]+:[a-zA-Z]+[0-9]+\b"""
        named_object_regex = r"""'(.*?)'"""
        excel_cell_objects = r"""\b([A-Z]+[0-9]+|[a-z]+[0-9]+)\b"""
        # conditions_operators = r"""(<)|(<=)|(>=)|(=)|(>)|(==)"""
        numerics = r'''\b(([0-9]+)|([0-9]+\.[0-9]*)|([0-9]+e\^([0-9]+)))\b'''
        range_objects = []
        named_objects = []
        cell_references = []
        conditions = []
        numbers_objects = []
        paragraph, formulas = preprocessing.extract_formulas(paragraph)
        def numeric_replacer(match):
            numbers_objects.append(match) 
            return "<Number>"
        def range_replacer(match):
            range_objects.append(match) 
            return "<Range>"
        def named_replacer(match):
            named_objects.append(match) 
            return "<Name>"
        def cell_replacer(match):
            cell_references.append(match) 
            return "<Cell>"
        def condition_replacer(match):
            conditions.append(match) 
            return "<Cond>"
        result = re.sub(range_regex, range_replacer, paragraph)
        result = re.sub(named_object_regex, named_replacer, result)
        result = re.sub(excel_cell_objects, cell_replacer, result)
        # result = re.sub(conditions_operators, condition_replacer, result)
        result = re.sub(numerics, numeric_replacer, result)
        return result, {"ranges":range_objects, "named_objects":named_objects, "cell_references": cell_references, "conditional_objects": conditions, "numerical_objects": numbers_objects, "formulas": formulas}
        
    
    @staticmethod
    def chunking(documents: str):
        document = Sentencizer(documents)
        chunks = document.sentences
        return chunks
    
    @staticmethod
    def normalize_sentence(sentence: str):
       normal = Normalizer()
       return normal.normalize_sentence(sentence)
   
    @staticmethod
    def spell_correct(words: str):
        return spell_correction(words)
    
    @staticmethod
    def contraction(words: str):
        return normalize_contractions(words)
    
    @staticmethod
    def tag_assembler(tokens: list[str]):
        assembled_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "<":
                if i < len(tokens) - 2 :
                    if tokens[i + 1] in ["range", "formula", "url", "hashtag", "email", "cond", "cell", "number", "date", "name"] and tokens[i + 2] == ">":
                        # tokens[i+1][0] = tokens[i+1][0].upper()
                        assembled_tokens.append("".join(tokens[i : i + 3]))
                        i += 3
                else:
                    assembled_tokens.append(tokens[i])
                    i += 1
            else:
                assembled_tokens.append(tokens[i])
                i += 1
        return assembled_tokens
            
            
    
    def preprocess(self, text: str):
        text, urls = preprocessing.extract_urls(text)
        text, email = preprocessing.extract_emails(text)
        text, dates = preprocessing.extract_dates(text)
        text, hashtags = preprocessing.extract_hashtags(text)
        text, excel_entities = preprocessing.extract_excel_entites(text)
        text = preprocessing.special_characters(text)
        sentences = preprocessing.chunking(text)
        data = {}
        for idx, sentence in enumerate(sentences):
            sentence = simplify_punctuation(sentence)
            tokens = Tokenizer(sentence).tokens
            data[idx] = preprocessing.tag_assembler(tokens)
        
        excel_entities["urls"] = urls
        excel_entities["emails"] = email
        excel_entities["dates"] = dates
        excel_entities["hashtags"] = hashtags
        excel_entities["sentences"] = data
        excel_entities["paragpraph"] = text
        return excel_entities
    
    
