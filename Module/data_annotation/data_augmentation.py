import pandas as pd
from tqdm import tqdm
from textgenie import TextGenie
import re

# Function to replace words in brackets
def replace_words_in_brackets(text):
    pattern = re.compile(r'<(.*?)>')
    result = pattern.sub(r'\1', text)
    return result

# Initialize TextGenie
geek = TextGenie("ramsrigouthamg/t5_paraphraser",'bert-base-uncased',"en_core_web_sm")

# Load and clean the dataset
data = pd.read_csv("balanced_data.csv")
data = data.dropna()

# Augment data
augmentations = []
for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    text = replace_words_in_brackets(row['prompt'])  # Assuming 'prompt' is the column name
    intent = row['intent']
    class_ = row['classes']
    augmentation = geek.augment_sent_t5(text, "paraphrase: ", 10)
    augmentations.extend([(sentence, intent, class_) for sentence in augmentation])

# Create new DataFrame and save to CSV
new_data = pd.DataFrame(augmentations, columns=["text", "intent", "classes"])
new_data.to_csv("Book1.csv", index=False)

# Fix warnings about deprecated functions in transformers (if any)
import transformers
transformers.logging.set_verbosity_error()

# Ensure spacy version compatibility
import spacy
spacy_version = spacy.__version__
print(f"spaCy version: {spacy_version}")

# If you encounter version compatibility issues, ensure you're using a compatible spaCy model version
