import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import pandas as pd
import numpy as np
from collections import Counter
from IPython.display import Markdown
import textwrap
import pickle
import ipdb
import google.generativeai as genai
import tensorflow as tf
#saves pickle file
def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

#loads pickle file
def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj
            
def key_swapper(dictionary: dict):
    swapped_dict = {value: key for key, value in dictionary.items()}
    return swapped_dict    
# Transfer labeled data for action name extractor. 
# The infile is original labeled data which can be use directly by action argument extractor
# e.g. infile = 'data/online_test/online_labeled_text.pkl' # is online-labeled data
#      outfile = 'data/online_test/online_labeled_text_data.pkl'
# PS: It's an a long
def transfer(infile, outfile):
    indata = load_pkl(infile)[-1]
    data = []
    log = {'wrong_last_sent': 0, 'act_reference_1': 0, 'related_act_reference_1': 0,'obj_reference_1': 0, 'non-obj_reference_1': 0}
    for i in range(len(indata)):
        words = []  # all words of a text
        sents = []  # all sentences of a text
        word2sent = {}  # transfer from a word index to its sentence index
        text_acts = []  # all actions of a text
        sent_acts = []  # actions of each sentence
        reference_related_acts = False
        for j in range(len(indata[i])):
            # labeling error: empty sentence
            if len(indata[i][j]) == 0:  
                # print('%s, len(indata[%d][%d]) == 0'%(self.domain, i, j))
                continue
            last_sent = indata[i][j]['last_sent']
            this_sent = indata[i][j]['this_sent']
            acts = indata[i][j]['acts']
            
            # labeling error: mis-matched sentences
            if j > 0 and len(last_sent) != len(indata[i][j-1]['this_sent']):
                b1 = len(last_sent)
                b2 = len(indata[i][j-1]['this_sent'])
                for k in range(len(acts)):
                    ai = acts[k]['act_idx']
                    new_act_type = acts[k]['act_type']
                    new_act_idx = ai - b1 + b2
                    new_obj_idxs = [[],[]]
                    for l in range(2):
                        for oi in acts[k]['obj_idxs'][l]:
                            if oi == -1:
                                new_obj_idxs[l].append(oi)
                            else:
                                new_obj_idxs[l].append(oi - b1 + b2)
                        assert len(new_obj_idxs[l]) == len(acts[k]['obj_idxs'][l])
                    new_related_acts = []
                    acts[k] = {'act_idx': new_act_idx, 'obj_idxs': new_obj_idxs,
                            'act_type': new_act_type, 'related_acts': new_related_acts}
                last_sent = indata[i][j-1]['this_sent']
                log['wrong_last_sent'] += 1

            sent = last_sent + this_sent
            last_sent_bias = len(last_sent)
            # pronoun resolution, find the source noun of a pronoun
            reference_obj_flag = False  
            tmp_acts = []
            for k in range(len(acts)):
                act_idx = acts[k]['act_idx']
                obj_idxs = acts[k]['obj_idxs']
                tmp_act_idx = act_idx - last_sent_bias
                if tmp_act_idx < 0:
                    log['act_reference_1'] += 1
                
                tmp_obj_idxs = [[],[]]
                for l in range(2):
                    for oi in obj_idxs[l]:
                        if oi == -1:
                            tmp_obj_idxs[l].append(oi)
                        else:
                            tmp_obj_idxs[l].append(oi - last_sent_bias)
                            if oi - last_sent_bias < 0:
                                reference_obj_flag = True
                    assert len(tmp_obj_idxs[l]) == len(obj_idxs[l])
                tmp_act_type = acts[k]['act_type']
                tmp_related_acts = []
                if len(acts[k]['related_acts']) > 0:
                    for idx in acts[k]['related_acts']:
                        tmp_related_acts.append(idx - last_sent_bias)
                        if idx - last_sent_bias < 0:
                            reference_related_acts = True
                            log['related_act_reference_1'] += 1
                    assert len(tmp_related_acts) == len(acts[k]['related_acts'])
                tmp_acts.append({'act_idx': tmp_act_idx, 'obj_idxs': tmp_obj_idxs,
                            'act_type': tmp_act_type, 'related_acts': tmp_related_acts})
            # assert len(tmp_acts) == len(acts)
            # labeling error: wrong word index in the first sentence 
            if j == 0:
                if reference_obj_flag:
                    log['obj_reference_1'] += 1
                    for ii in range(len(words), len(words)+len(last_sent)):
                        word2sent[ii] = len(sents)
                    words.extend(last_sent)
                    sents.append(last_sent)
                    sent_acts.append({})
                else:
                    if len(last_sent) > 0:
                        log['non-obj_reference_1'] += 1
                        last_sent = []
                        last_sent_bias = len(last_sent)
                        sent = last_sent + this_sent
                        acts = tmp_acts

            
            for ii in range(len(words), len(words)+len(this_sent)):
                word2sent[ii] = len(sents)
            all_word_bias = len(words)
            words.extend(this_sent)
            sents.append(this_sent)
            sent_acts.append(acts)
            all_acts_of_cur_sent = update_acts(words, sent, last_sent_bias, all_word_bias, tmp_acts)
            text_acts.extend(all_acts_of_cur_sent)

        # assert len(word2sent) == len(words)
        # assert len(sents) == len(sent_acts)
        data.append({'words': words, 'acts': text_acts, 'sent_acts': sent_acts,
                    'sents': sents, 'word2sent': word2sent})
    upper_bound, lower_bound = compute_context_len(data)
    print('\nupper_bound: {}\tlower_bound: {}\nlog history: {}\n'.format(upper_bound, lower_bound, log))
    save_pkl(data, outfile)



def update_acts(words, sent, last_sent_bias, all_word_bias, tmp_acts):
    # all indices of the words in the current sentences need to add a last_sent_bias
    all_acts_of_cur_sent = []
    for k in range(len(tmp_acts)):
        act_idx = tmp_acts[k]['act_idx']
        obj_idxs = tmp_acts[k]['obj_idxs']
        text_act_idx = act_idx + all_word_bias
        # labeling error: mis-matched word index
        if sent[act_idx + last_sent_bias] != words[act_idx + all_word_bias]:
            print(sent[act_idx + last_sent_bias], words[act_idx + all_word_bias])
        text_obj_idxs = [[],[]]
        for l in range(2):
            for oi in obj_idxs[l]:
                if oi == -1:
                    text_obj_idxs[l].append(-1)
                else:
                    text_obj_idxs[l].append(oi + all_word_bias)
                    if sent[oi + last_sent_bias] != words[oi + all_word_bias]:
                        ipdb.set_trace()
                        print(sent[oi + last_sent_bias], words[oi + all_word_bias])
            # assert len(text_obj_idxs[l]) == len(obj_idxs[l])
        text_act_type = tmp_acts[k]['act_type']
        text_related_acts = []
        if len(tmp_acts[k]['related_acts']) > 0:
            for idx in tmp_acts[k]['related_acts']:
                text_related_acts.append(idx + all_word_bias)
            # assert len(text_related_acts) == len(tmp_acts[k]['related_acts'])
        acts = {'act_idx': text_act_idx, 'obj_idxs': text_obj_idxs,
                'act_type': text_act_type, 'related_acts': text_related_acts}
        all_acts_of_cur_sent.append(acts)
    return all_acts_of_cur_sent


# Compute the length of context for action argument extractor
# the upper_bound/lower_bound indicate how far/near between the action name and its arguments
# the difference between them is used to control the context_len
# e.g. context_len = 2 * upper_bound
def compute_context_len(data):
    upper_bound = 0
    lower_bound = 0
    for d in data:
        for n in range(len(d['acts'])):
            act = d['acts'][n]['act_idx']
            objs = d['acts'][n]['obj_idxs']
            for l in range(2):
                for obj in objs[l]:
                    if obj == -1:
                        continue
                    if obj - act < lower_bound:
                        lower_bound = obj - act
                    if obj - act > upper_bound:
                        upper_bound = obj - act
    return upper_bound, lower_bound




def ten_fold_split_ind(num_data, fname, k, random=True):
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = num_data/k
    indices = []

    if random:
        tmp_inds = np.arange(num_data)
        np.random.shuffle(tmp_inds)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_inds[i*n: ])
            else:
                indices.append(tmp_inds[i*n: (i+1)*n])
    else:
        for i in range(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices



def index2data(indices, data):
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    if type(data) == dict:
        keys = data.keys()
        print('data.keys: {}'.format(keys))
        num_data = len(data[keys[0]])
        for i in range(len(indices)):
            valid_data = {}
            train_data = {}
            for k in keys:
                valid_data[k] = []
                train_data[k] = []
            for ind in range(num_data):
                for k in keys:
                    if ind in indices[i]:
                        valid_data[k].append(data[k][ind])
                    else:
                        train_data[k].append(data[k][ind])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)
    else:
        num_data = len(data)
        for i in range(len(indices)):
            valid_data = []
            train_data = []
            for ind in range(num_data):
                if ind in indices[i]:
                    valid_data.append(data[ind])
                else:
                    train_data.append(data[ind])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)

    return folds


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def action_generation(Act_type: str):    
    from IPython.display import display
    from IPython.display import Markdown
    genai.configure(api_key="AIzaSyBYYJwGBDq-Crti9zwuobkVtKUml2W34jw")
    model = genai.GenerativeModel('gemini-pro')
    action  = model.generate_content(Act_type).candidates[0].content.parts

def excel_column_to_index(column_str: str):
    base = ord('A') - 1  # Base for numerical conversion
    result = 0
    for char in column_str.upper():
        if not char.isalpha():
            raise ValueError("Invalid column string: {}".format(column_str))
        result = result * 26 + (ord(char) - base)
    return result    
    
import math
from collections import Counter


#computes point mutual index between every action and every word in the corpus
def compute_pmi(corpus: list[str], actions: list[str]):
    total_docs = len(corpus)
    action_counts = Counter(actions)
    corpus_word_counts = Counter(word for doc in corpus for word in doc.split())

    pmi_scores = {}

    for action in action_counts:
        p_action = action_counts[action] / total_docs

        for word in corpus_word_counts:
            p_word = corpus_word_counts[word] / total_docs
            p_joint = (sum(1 for doc in corpus if action in doc and word in doc) + 1) / total_docs  # Add-one smoothing

            pmi = math.log2(p_joint / (p_action * p_word))

            if pmi > 0:
                pmi_scores[(action, word)] = pmi

    return pmi_scores
   
   
#==========================================================
#==================== dataset functions ===================
#==========================================================

def load_dataset_ffn(file_path):
    df = pd.read_excel(file_path)
    prompts = df['prompt'].tolist()
    contexts = df['context'].tolist()
    outputs = df['output'].tolist()
    inputs = [prompt + ' ' + context for prompt, context in zip(prompts, contexts)]
    return inputs, outputs
 
    
def load_dataset(file_path, sheet_name, datatypes):
    df = pd.read_excel(file_path,sheet_name)
    data = []
    for datatype in datatypes:
        data.append({datatype: df[datatype].tolist()})
    return data


#==========================================================
#==================== model functions =====================
#==========================================================

def linear(x: tf.Tensor, output_size: int, name: str, activation = tf.nn.relu):
    with tf.compat.v1.variable_scope(name):
        weights = tf.compat.v1.get_variable("w", [x.shape[1], output_size], tf.float32, initializer=tf.compat.v1.initializers.truncated_normal(0, 0.1)) # weight initializers
        bias = tf.compat.v1.get_variable("b", [output_size], initializer=tf.compat.v1.constant_initializer(0.1))
        output = tf.nn.bias_add(tf.matmul(x, weights), bias)
        output = activation(output)
        return output, weights, bias
    
def conv2d(x: tf.Tensor , output_dim: int, kernel_size, stride, initializer, activation_fn = tf.nn.relu, name: str = '', padding='VALID'):
    with tf.compat.v1.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_size = [kernel_size[0], kernel_size[1], x.shape[-1], output_dim]
        weights = tf.compat.v1.get_variable('w', kernel_size, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, weights, stride, padding)
        bias = tf.compat.v1.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
        out = tf.nn.bias_add(conv, bias)
        out = activation_fn(out)
        return out, weights, bias
    
def max_pooling(x: tf.Tensor, kernel_size, stride, name: str = '',padding='VALID'):
    with tf.compat.v1.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_size = [1, kernel_size[0], kernel_size[1], 1]
        return tf.nn.max_pool(x, kernel_size, stride, padding)

def avg_pooling(x: tf.Tensor, kernel_size, stride, name:str = '',padding='VALID'):
    with tf.compat.v1.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_size = [1, kernel_size[0], kernel_size[1], 1]
        return tf.nn.avg_pool(x, kernel_size, stride, padding)
