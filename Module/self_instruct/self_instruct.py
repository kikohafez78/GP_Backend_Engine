import pandas as pd
from textgenie import TextGenie
# from Module.preprocessing.preprocessing import Sentencizer
import torch
torch.set_warn_always(False)

class self_instruct:
    model = textgenie = TextGenie("ramsrigouthamg/t5_paraphraser",'bert-base-uncased', device = "cuda")
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        
    def generate_parphrases_based_on_s(self, text: str): #based on alone sentences
        if self.config["mode"] == "sentences":
            # sentences = Sentencizer(text).sentences
            # generated_outputs = {}
            # for sentence in sentences:
            #     generated_outputs[sentence] = self.textgenie.augment_sent_t5(sentence,"paraphrase: ",self.config["n_predictions"])
            # return generated_outputs
            pass
        else:
            generated_outputs = self.textgenie.augment_sent_t5(text, "paraphrase: ", self.config["n_predictions"])
            return generated_outputs
    
    def augment_auto(self, prompt: list[str], intents: list[str]):
        augmented_data = {}
        for sentence, intent in zip(prompt, intents):
            augmented_data[(sentence, intent)] = self.generate_parphrases_based_on_s(sentence)
        return augmented_data
    
    def augment_dataset_n_save(self, prompts: list[list[str]], intents: list[str]):
        augmented_prompts = {}
        intent_len = len(intent_len)
        for i in range(len(prompts)):
            plan_length = len(prompts[i])
            augmented_prompts[f"plan{i}"] = self.augment_auto(prompts[i], intents[i:(i+plan_length) if (i + plan_length) < intent_len else -1])
        return augmented_prompts
    
    
    
            


from self_Instruct_config import get_self_instruct


config = get_self_instruct()
model = self_instruct("a7a", config)
model.generate_parphrases_based_on_s("create a new sheet called 'hero'")