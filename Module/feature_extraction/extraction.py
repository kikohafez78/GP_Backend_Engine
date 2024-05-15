from text_feature_extraction import ConsecutiveNPChunker, get_tfidf_given_file, tags_since_dt, text_features
from ExcelToJson import excel_information_extractor




class Feature_Extraction:
    def __init__(self, mode: int = 1):
        self.mode = mode
        return 
    