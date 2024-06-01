import nltk
import spacy 
import stanza
import re
from tfidf import TFIDF
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
import sys
import os 
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def replace_and_store(match):
    content = match.group(1).split()[0]
    bracket = match.group(1).split()[1]
    contents_list.append(content)
    brackets_list.append(bracket)
    return '*'

def conll_table(string, lemmas, ner, sent_count):
    stared_string = re.sub(r'\(([^()]+)\)', replace_and_store, string)

    print(brackets_list)
    stared_string = stared_string.replace('ROOT', 'TOP')
    stared_string = ''.join(stared_string.split())
    stared_string = re.sub(r'\*(\*)', r'* \1', stared_string)
    stared_string = re.sub(r'\*\*', r'* *', stared_string)
    stared_string = re.sub(r'\*(\()', r'* \1', stared_string)
    stared_string = re.sub(r'\)(\*)', r') \1', stared_string)
    stared_string = re.sub(r'\)(\()', r') \1', stared_string)

    tree1 = stared_string.split()
    print(tree1)

    table = []
    c = 0
    for i in range(len(contents_list)):
        table.append(['wb/a2e/00/a2e_0010', '0', c, brackets_list[i], contents_list[i], tree1[i], lemmas[i], '-', '-', '-', ner[i], '-'])
        c += 1
        if brackets_list[i] == ".":
            c = 0
      
    return table

def wrapper(doc):
    global contents_list, brackets_list
    contents_list = []
    brackets_list = []

    tree_dict = {}
    table_list = []
    lemmas = []
    ner = []

    for sent in doc.sentences:
        cons = sent.constituency
        tree_dict[sent] = str(cons)
        for token in sent.tokens:
            if token.ner == 'O':
                ner.append('*')
            else:
                ner.append('*')

        for word in sent.words:
            if word.upos == "VERB":
                lemmas.append(word.lemma)
            else:
                lemmas.append('-')

    combined_tree = ''
    for sent in doc.sentences:
       combined_tree += ' ' + ' '.join(tree_dict[sent].split())

    table_list.append(conll_table(combined_tree, lemmas, ner, len(doc.sentences)))

    return table_list



def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
    return '+'.join(sorted(tags))

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
         prevword, prevpos = "<START>", "<START>"
    else:
         prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
         nextword, nextpos = sentence[i+1]
    return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos,
             "prevpos+pos": "%s+%s" % (prevpos, pos),
             "pos+nextpos": "%s+%s" % (pos, nextpos),
              "tags-since-dt": tags_since_dt(sentence, i)} 


class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

class text_features:
    spacy_tool = spacy.load('en_core_web_sm')
    stanza_tool = stanza.Pipeline('en', processors='tokenize,pos,lemma,constituency,ner,mwt')
    tool1 = ConsecutiveNPChunker
    tool2 = ConsecutiveNPChunkTagger
    temporal_connectives = {
    "before": "precedence",
    "after": "succession",
    "while": "simultaneity",
    "until": "termination",
    "since": "causality",
    "when": "causality",
    "as soon as": "immediate succession",
    "whenever": "repetition",
    "once": "single occurrence",
    "by the time": "deadline",
    "during": "duration",
    "then": "succession",
    "later": "succession",
    "earlier": "precedence"
    }
    with open(os.path.join(os.getcwd(), "Module\\feature_extraction\\temporal.txt")) as f:
        temporal = f.readlines()
    def __init__(self, tokens: list[list[str]]):
        self.tokens = tokens
        self.pos_tags = [nltk.pos_tag(token) for sentence in tokens for token in sentence if token  not in ["<EOS>","<SOS>"]]
        self.embeddings = []
        self.ner = [nltk.ne_chunk(sent) for sent in self.pos_tags]
        for sentence in tokens:
            sent = self.tool(" ".join(sentence))
            self.embeddings.append([token.vector for token in sent])
            
    def get_tfidf(self, text):
        return TFIDF(text)
    
    

    def extract_temporal_connectives(self, paragraph: str):
        doc = self.spacy_tool(paragraph)
        connectives_info = []

        # Iterate through sentences in the paragraph
        for sent in doc.sents:
            sent_text = sent.text
            sent_start_idx = sent.start_char
            sent_end_idx = sent.end_char

            # Check for temporal connectives within the sentence
            for connective, connective_type in self.temporal_connectives.items():
                if connective in sent_text:
                    start_idx = sent_text.index(connective)
                    end_idx = start_idx + len(connective)
                    # Calculate the absolute indices within the paragraph
                    abs_start_idx = sent_start_idx + start_idx
                    abs_end_idx = sent_start_idx + end_idx

                    # Determine the text before and after the connective
                    before_text = sent_text[:start_idx].strip()
                    after_text = sent_text[end_idx:].strip()

                    connectives_info.append({
                        "connective": connective,
                        "type": connective_type,
                        "start_idx": abs_start_idx,
                        "end_idx": abs_end_idx,
                        "precedence": {
                            "before": before_text,
                            "after": after_text
                        }
                    })

            return connectives_info
            
    def conll_features(self, text: str):
        return wrapper(text)[0]
        
    
    
                
                
