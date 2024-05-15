from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import six
import tensorflow as tf
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import keras as ks
import transformers as optimus
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.layers import Input, Dropout, InputSpec, Conv1D, Add
import keras.backend as K
import torch
import torch.nn as nn
tf.config.run_functions_eagerly(True)



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def sparse_categorical_crossentropy(y_true, y_pred):
    loss = ks.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)


py_any = K.any
ndim = K.ndim
bert_path = "uncased_L-12_H-768_A-12"
bert_ckpt_file = os.path.join(bert_path, 'bert_model.ckpt')
bert_config_file = os.path.join(bert_path, 'bert_config.json')
vocab_file = os.path.join(bert_path, 'vocab.txt')
class IntentDetectionData:
  DATA_COLUMN = "text"
  LABEL_COLUMN = "intent"

  def __init__(self, train, test, tokenizer: optimus.BertTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    train, test = map(lambda df: df.reindex(df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index), [train, test])
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.asarray(x, dtype = "object"), np.array(y, dtype = "object")

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.asarray(input_ids, dtype = "object"))
    return np.asarray(x, dtype = "object")
def batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if axes is None:
        axes = [x_ndim - 1, y_ndim - 2]
    if py_any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out
def shape_list(x):
    if K.backend() != 'theano':
        tmp = K.int_shape(x)
    else:
        tmp = x.shape
    tmp = list(tmp)
    tmp[0] = -1
    return tmp


def split_heads(x, n: int, k: bool = False): 
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    new_x = K.reshape(x, new_x_shape)
    return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])


def merge_heads(x):
    new_x = K.permute_dimensions(x, [0, 2, 1, 3])
    x_shape = shape_list(new_x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return K.reshape(new_x, new_x_shape)


def scaled_dot_product_attention_tf(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):
    w = batch_dot(q, k)  
    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
    if attn_mask is not None:
        w = attn_mask * w + (1.0 - attn_mask) * neg_inf
    w = K.softmax(w)
    w = Dropout(attention_dropout)(w)
    return batch_dot(w, v)  


def scaled_dot_product_attention_th(q, k, v, attn_mask, attention_dropout: float, neg_inf: float):
    w = theano_matmul(q, k)
    w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
    if attn_mask is not None:
        attn_mask = K.repeat_elements(attn_mask, shape_list(v)[1], 1)
        w = attn_mask * w + (1.0 - attn_mask) * neg_inf
    w = K.T.exp(w - w.max()) / K.T.exp(w - w.max()).sum(axis=-1, keepdims=True)
    w = Dropout(attention_dropout)(w)
    return theano_matmul(w, v)


def multihead_attention(x, attn_mask, n_head: int, n_state: int, attention_dropout: float, neg_inf: float):
    _q, _k, _v = x[:, :, :n_state], x[:, :, n_state:2 * n_state], x[:, :, -n_state:]
    q = split_heads(_q, n_head)  
    k = split_heads(_k, n_head, k=True)  
    v = split_heads(_v, n_head)  
    if K.backend() == 'tensorflow':
        a = scaled_dot_product_attention_tf(q, k, v, attn_mask, attention_dropout, neg_inf)
    else:
        a = scaled_dot_product_attention_th(q, k, v, attn_mask, attention_dropout, neg_inf)
    return merge_heads(a)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))

def theano_matmul(a, b, _left=False):
    assert a.ndim == b.ndim
    ndim = a.ndim
    assert ndim >= 2
    if _left:
        b, a = a, b
    if ndim == 2:
        return K.T.dot(a, b)
    else:
        if a.broadcastable[0] and not b.broadcastable[0]:
            output, _ = K.theano.scan(theano_matmul, sequences=[b], non_sequences=[a[0], 1])
        elif b.broadcastable[0] and not a.broadcastable[0]:
            output, _ = K.theano.scan(theano_matmul, sequences=[a], non_sequences=[b[0]])
        else:
            output, _ = K.theano.scan(theano_matmul, sequences=[a, b])
        return output
class MultiHeadAttention(ks.layers.Layer):
    def __init__(self, n_head: int, n_state: int, attention_dropout: float, use_attn_mask: bool, neg_inf: float,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_head = n_head
        self.n_state = n_state
        self.attention_dropout = attention_dropout
        self.use_attn_mask = use_attn_mask
        self.neg_inf = neg_inf

    def compute_output_shape(self, input_shape):
        x = input_shape[0] if self.use_attn_mask else input_shape
        return x[0], x[1], x[2] // 3

    def call(self, inputs, **kwargs):
        x = inputs[0] if self.use_attn_mask else inputs
        attn_mask = inputs[1] if self.use_attn_mask else None
        return multihead_attention(x, attn_mask, self.n_head, self.n_state, self.attention_dropout, self.neg_inf)

    def get_config(self):
        config = {
            'n_head': self.n_head,
            'n_state': self.n_state,
            'attention_dropout': self.attention_dropout,
            'use_attn_mask': self.use_attn_mask,
            'neg_inf': self.neg_inf,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Zeros():
    return 

def Ones():
    return 

class LayerNormalization(ks.layers.Layer):
    def __init__(self, eps: float = 1e-5, **kwargs) -> None:
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x, **kwargs):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / K.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Gelu(ks.layers.Layer):
    def __init__(self, accurate: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.accurate = accurate

    def call(self, inputs, **kwargs):
        if not self.accurate:
            return gelu(inputs)
        if K.backend() == 'tensorflow':
            erf = K.tf.erf
        else:
            erf = K.T.erf
        return inputs * 0.5 * (1.0 + erf(inputs / np.sqrt(2.0)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'accurate': self.accurate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
def _get_pos_encoding_matrix(max_len: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(max_len)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class BertEmbedding(ks.layers.Layer):
    def __init__(self, output_dim: int = 768, dropout: float = 0.1, vocab_size: int = 30000,
                 max_len: int = 512, trainable_pos_embedding: bool = True, use_one_dropout: bool = False,
                 use_embedding_layer_norm: bool = False, layer_norm_epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.use_one_dropout = use_one_dropout
        self.output_dim = output_dim
        self.dropout = dropout
        self.vocab_size = vocab_size

        # Bert ks uses two segments for next-sentence classification task
        self.segment_emb = ks.layers.Embedding(2, output_dim, input_length=max_len,
                                                  name='SegmentEmbedding')

        self.trainable_pos_embedding = trainable_pos_embedding
        if not trainable_pos_embedding:
            self.pos_emb = ks.layers.Embedding(max_len, output_dim, trainable=False, input_length=max_len,
                                                  name='PositionEmbedding',
                                                  weights=[_get_pos_encoding_matrix(max_len, output_dim)])
        else:
            self.pos_emb = ks.layers.Embedding(max_len, output_dim, input_length=max_len, name='PositionEmbedding')

        self.token_emb = ks.layers.Embedding(vocab_size, output_dim, input_length=max_len, name='TokenEmbedding')
        self.embedding_dropout = ks.layers.Dropout(dropout, name='EmbeddingDropOut')
        self.add_embeddings = ks.layers.Add(name='AddEmbeddings')
        self.use_embedding_layer_norm = use_embedding_layer_norm
        if self.use_embedding_layer_norm:
            self.embedding_layer_norm = LayerNormalization(layer_norm_epsilon)
        else:
            self.embedding_layer_norm = None
        self.layer_norm_epsilon = layer_norm_epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim

    def get_config(self):
        config = {
            'max_len': self.max_len,
            'use_one_dropout': self.use_one_dropout,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'vocab_size': self.vocab_size,
            'trainable_pos_embedding': self.trainable_pos_embedding,
            'embedding_layer_norm': self.use_embedding_layer_norm,
            'layer_norm_epsilon': self.layer_norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __call__(self, inputs, **kwargs):
        tokens, segment_ids, pos_ids = inputs
        segment_embedding = self.segment_emb(segment_ids)
        pos_embedding = self.pos_emb(pos_ids)
        token_embedding = self.token_emb(tokens)
        if self.use_one_dropout:
            summation = self.add_embeddings([segment_embedding, pos_embedding, token_embedding])
            if self.embedding_layer_norm:
                summation = self.embedding_layer_norm(summation)
            return self.embedding_dropout(summation)
        summation = self.add_embeddings(
            [self.embedding_dropout(segment_embedding), self.embedding_dropout(pos_embedding),
             self.embedding_dropout(token_embedding)])
        if self.embedding_layer_norm:
            summation = self.embedding_layer_norm(summation)
        return summation
class MultiHeadSelfAttention:
    def __init__(self, n_state: int, n_head: int, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float) -> None:
        assert n_state % n_head == 0
        self.c_attn = Conv1D(3 * n_state, 1, name='layer_{}/c_attn'.format(layer_id))
        self.attn = MultiHeadAttention(n_head, n_state, attention_dropout, use_attn_mask,
                                       neg_inf, name='layer_{}/self_attention'.format(layer_id))
        self.c_attn_proj = Conv1D(n_state, 1, name='layer_{}/c_attn_proj'.format(layer_id))

    def __call__(self, x, mask):
        output = self.c_attn(x)
        output = self.attn(output) if mask is None else self.attn([output, mask])
        return self.c_attn_proj(output)


class PositionWiseFF:
    def __init__(self, n_state: int, d_hid: int, layer_id: int, accurate_gelu: bool) -> None:
        self.c_fc = Conv1D(d_hid, 1, name='layer_{}/c_fc'.format(layer_id))
        self.activation = Gelu(accurate=accurate_gelu, name='layer_{}/gelu'.format(layer_id))
        self.c_ffn_proj = Conv1D(n_state, 1, name='layer_{}/c_ffn_proj'.format(layer_id))

    def __call__(self, x):
        output = self.activation(self.c_fc(x))
        return self.c_ffn_proj(output)


class EncoderLayer:
    def __init__(self, n_state: int, n_head: int, d_hid: int, residual_dropout: float, attention_dropout: float,
                 use_attn_mask: bool, layer_id: int, neg_inf: float, ln_epsilon: float, accurate_gelu: bool) -> None:
        self.Inputspec = InputSpec(ndim=3)
        self.attention = MultiHeadSelfAttention(n_state, n_head, attention_dropout, use_attn_mask, layer_id, neg_inf)
        self.drop1 = Dropout(residual_dropout, name='layer_{}/ln_1_drop'.format(layer_id))
        self.add1 = Add(name='layer_{}/ln_1_add'.format(layer_id))
        self.ln1 = LayerNormalization(ln_epsilon, name='layer_{}/ln_1'.format(layer_id))
        self.ffn = PositionWiseFF(n_state, d_hid, layer_id, accurate_gelu)
        self.drop2 = Dropout(residual_dropout, name='layer_{}/ln_2_drop'.format(layer_id))
        self.add2 = Add(name='layer_{}/ln_2_add'.format(layer_id))
        self.ln2 = LayerNormalization(ln_epsilon, name='layer_{}/ln_2'.format(layer_id))
        print("problem 1")

    def __call__(self, x, mask):
        print("problem 2")
        print(x)
        a = self.attention(x, mask)
        print("problem 3")
        n = self.ln1(self.add1([x, self.drop1(a)]))
        print("problem 4")
        f = self.ffn(n)
        return self.ln2(self.add2([n, self.drop2(f)]))


def create_transformer(embedding_dim: int = 768, embedding_dropout: float = 0.1, vocab_size: int = 30000,
                       max_len: int = 512, trainable_pos_embedding: bool = True, num_heads: int = 12,
                       num_layers: int = 12, attention_dropout: float = 0.1, use_one_embedding_dropout: bool = False,
                       d_hid: int = 768 * 4, residual_dropout: float = 0.1, use_attn_mask: bool = True,
                       embedding_layer_norm: bool = False, neg_inf: float = -1e9, layer_norm_epsilon: float = 1e-5,
                       accurate_gelu: bool = False) -> ks.Model:
    tokens = Input(batch_shape=(None, max_len), name='token_input', dtype='int32')
    segment_ids = Input(batch_shape=(None, max_len), name='segment_input', dtype='int32')
    pos_ids = Input(batch_shape=(None, max_len), name='position_input', dtype='int32')
    attn_mask = Input(batch_shape=(None, 1, max_len, max_len), name='attention_mask_input',
                      dtype=K.floatx()) if use_attn_mask else None
    inputs = [tokens, segment_ids, pos_ids]
    embedding_layer = BertEmbedding(embedding_dim, embedding_dropout, vocab_size, max_len, trainable_pos_embedding,
                                    use_one_embedding_dropout, embedding_layer_norm, layer_norm_epsilon)
    x = embedding_layer(inputs)
    for i in range(num_layers):
        x = EncoderLayer(embedding_dim, num_heads, d_hid, residual_dropout,
                         attention_dropout, use_attn_mask, i, neg_inf, layer_norm_epsilon, accurate_gelu)(x, attn_mask)
    if use_attn_mask:
        inputs.append(attn_mask)
    return ks.Model(inputs=inputs, outputs=[x], name='Transformer')



class BertConfig(object):

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
def get_bert_weights_for_keras_model(check_point, max_len, model, tf_var_names):
    keras_weights = [np.zeros(w.shape) for w in model.weights]
    keras_weights_set = []

    for var_name, _ in tf_var_names:
        qkv, unsqueeze, w_id = _get_tf2keras_weights_name_mapping(var_name)
        if w_id is None:
            print('not mapped: ', var_name)  # TODO pooler, cls/predictions, cls/seq_relationship
        else:
            print(var_name, ' -> ', model.weights[w_id].name)
            keras_weights_set.append(w_id)
            keras_weight = keras_weights[w_id]
            tensorflow_weight = check_point.get_tensor(var_name)
            keras_weights[w_id] = _set_keras_weight_from_tf_weight(max_len, tensorflow_weight, keras_weight, qkv, unsqueeze, w_id)

    keras_layer_not_set = set(list(range(len(keras_weights)))) - set(keras_weights_set)
    assert len(keras_layer_not_set) == 0, 'Some weights were not set!'

    return keras_weights


def _set_keras_weight_from_tf_weight(max_len, tensorflow_weight, keras_weight, qkv, unsqueeze, w_id):
    if qkv is None:
        if w_id == 1:  # pos embedding
            keras_weight[:max_len, :] = tensorflow_weight[:max_len, :] if not unsqueeze else tensorflow_weight[None, :max_len, :]

        elif w_id == 2:  # word embedding
            keras_weight = tensorflow_weight
        else:
            keras_weight[:] = tensorflow_weight if not unsqueeze else tensorflow_weight[None, ...]
    else:
        p = {'q': 0, 'k': 1, 'v': 2}[qkv]
        if keras_weight.ndim == 3:
            dim_size = keras_weight.shape[1]
            keras_weight[0, :, p * dim_size:(p + 1) * dim_size] = tensorflow_weight if not unsqueeze else tensorflow_weight[None, ...]
        else:
            dim_size = keras_weight.shape[0] // 3
            keras_weight[p * dim_size:(p + 1) * dim_size] = tensorflow_weight

    return keras_weight


def _get_tf2keras_weights_name_mapping(var_name):
    w_id = None
    qkv = None
    unsqueeze = False

    var_name_splitted = var_name.split('/')
    if var_name_splitted[1] == 'embeddings':
        w_id = _get_embeddings_name(var_name_splitted)

    elif var_name_splitted[2].startswith('layer_'):
        qkv, unsqueeze, w_id = _get_layers_name(var_name_splitted)

    return qkv, unsqueeze, w_id


def _get_layers_name(var_name_splitted):
    first_vars_size = 5
    w_id = None
    qkv = None
    unsqueeze = False

    layer_number = int(var_name_splitted[2][len('layer_'):])
    if var_name_splitted[3] == 'attention':
        if var_name_splitted[-1] == 'beta':
            w_id = first_vars_size + layer_number * 12 + 5
        elif var_name_splitted[-1] == 'gamma':
            w_id = first_vars_size + layer_number * 12 + 4
        elif var_name_splitted[-2] == 'dense':
            if var_name_splitted[-1] == 'bias':
                w_id = first_vars_size + layer_number * 12 + 3
            elif var_name_splitted[-1] == 'kernel':
                w_id = first_vars_size + layer_number * 12 + 2
                unsqueeze = True
            else:
                raise ValueError()
        elif var_name_splitted[-2] == 'key' or var_name_splitted[-2] == 'query' or var_name_splitted[-2] == 'value':
            w_id = first_vars_size + layer_number * 12 + (0 if var_name_splitted[-1] == 'kernel' else 1)
            unsqueeze = var_name_splitted[-1] == 'kernel'
            qkv = var_name_splitted[-2][0]
        else:
            raise ValueError()
    elif var_name_splitted[3] == 'intermediate':
        if var_name_splitted[-1] == 'bias':
            w_id = first_vars_size + layer_number * 12 + 7
        elif var_name_splitted[-1] == 'kernel':
            w_id = first_vars_size + layer_number * 12 + 6
            unsqueeze = True
        else:
            raise ValueError()
    elif var_name_splitted[3] == 'output':
        if var_name_splitted[-1] == 'beta':
            w_id = first_vars_size + layer_number * 12 + 11
        elif var_name_splitted[-1] == 'gamma':
            w_id = first_vars_size + layer_number * 12 + 10
        elif var_name_splitted[-1] == 'bias':
            w_id = first_vars_size + layer_number * 12 + 9
        elif var_name_splitted[-1] == 'kernel':
            w_id = first_vars_size + layer_number * 12 + 8
            unsqueeze = True
        else:
            raise ValueError()
    return qkv, unsqueeze, w_id


def _get_embeddings_name(parts):
    n = parts[-1]
    if n == 'token_type_embeddings':
        w_id = 0
    elif n == 'position_embeddings':
        w_id = 1
    elif n == 'word_embeddings':
        w_id = 2
    elif n == 'gamma':
        w_id = 3
    elif n == 'beta':
        w_id = 4
    else:
        raise ValueError()
    return w_id
def create_model(base_location: str = '../uncased_L-12_H-768_A-12',
                     use_attn_mask: bool = True, max_len: int = 512) -> ks.Model:
    bert_config = BertConfig.from_json_file(base_location + '/bert_config.json')
    print(bert_config.__dict__)
    init_checkpoint = base_location + '/bert_model.ckpt'
    var_names = tf.train.list_variables(init_checkpoint)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, use_attn_mask=use_attn_mask,
                               vocab_size=bert_config.vocab_size, accurate_gelu=True, layer_norm_epsilon=1e-12, max_len=max_len,
                               use_one_embedding_dropout=True, d_hid=bert_config.intermediate_size,
                               embedding_dim=bert_config.hidden_size, num_layers=bert_config.num_hidden_layers,
                               num_heads=bert_config.num_attention_heads,
                               residual_dropout=bert_config.hidden_dropout_prob,
                               attention_dropout=bert_config.attention_probs_dropout_prob)
    weights = get_bert_weights_for_keras_model(check_point, max_len, model, var_names)
    model.set_weights(weights)
    return model


class Attention(nn.Module):

    def __init__(self, dimensions):
        super(Attention, self).__init__()
        self.dimensions = dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, attention_mask):
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        if attention_mask is not None:
            attention_mask = torch.unsqueeze(attention_mask, 2)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        attention_weights = self.softmax(attention_scores)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        output = self.linear_out(combined)
        output = self.tanh(output)
        return output, attention_weights
    
    
class classifier:
    def __init__(self, 
                 model_name: str,
                 max_seq_len: int,
                 classes: list[str],
                 model_path: str = None,
                 dropout: float = 0.1,
                 tokenizer: str = None,
                 extended: bool = False,
                 slot_classes: list[str] = None,
                 multi: bool = False
                ):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.classes = classes
        self.model_path = model_path
        self.dropout = dropout
        self.extended = extended
        self.tokenizer = tokenizer
        self.slot_classes = slot_classes
        self.multi = multi 
        
    def build(self):
        if not self.tokenizer:
            self.tokenizer = optimus.BertTokenizer.from_pretrained(self.model_path)
        elif self.tokenizer.lower() == "bert":
            self.tokenizer = optimus.BertTokenizer.from_pretrained(self.model_path)
        elif self.tokenizer.lower() == "albert":
            self.tokenizer = optimus.AlbertTokenizer.from_pretrained()
        elif self.tokenizer.lower() == "roberta":
            self.tokenizer = optimus.RobertaTokenizer.from_pretrained()
            
        if not self.extended:
            bert = optimus.TFBertModel.from_pretrained(self.model_path)
            input_ids = ks.layers.Input(shape=(self.max_seq_len,), dtype='int32')
            reshape_layer = ks.layers.Lambda(lambda x: x[:,0,:])
            dropout_layer_1 = ks.layers.Dropout(self.dropout)
            hidden_out = ks.layers.Dense(768, activation='softmax', kernel_initializer="random_normal")
            dropout_layer_2 = ks.layers.Dropout(self.dropout)
            class_out = ks.layers.Dense(len(self.classes), activation='softmax', kernel_initializer="random_normal")
            # class_reshape = ks.layers.Reshape((7,))
            bert_out = bert(input_ids)
            reshape_lambda = reshape_layer(bert_out[0])
            dropout_1 = dropout_layer_1(reshape_lambda)
            hidden_out = hidden_out(dropout_1)
            dropout_2 = dropout_layer_2(hidden_out)
            class_out = class_out(dropout_2)
            # class_out = class_reshape(class_out)
            self.model = ks.models.Model(inputs = input_ids, outputs = class_out)
            self.model.build(self.max_seq_len)
            print(self.model.output_shape)
        else: 
            
            bert = optimus.BertModel.from_pretrained(self.model_name)
            input_ids = ks.layers.Input(shape=(self.max_seq_len,), dtype='int32')
            reshape_layer = ks.layers.Lambda(lambda x: x[:,0,:])
            dropout_layer_1 = ks.layers.Dropout(self.dropout)
            hidden_out = ks.layers.Dense(768, activation='softmax', kernel_initializer="random_normal")
            dropout_layer_2 = ks.layers.Dropout(self.dropout)
            intent_classifier = ks.layers.Dense(len(self.classes), activation='softmax', kernel_initializer="random_normal")
            slot_classifier = ks.layers.Dense(len(self.slot_classes), activation='softmax', kernel_initializer="random_normal")
            
            bert_out = bert(input_ids)
            reshape_lambda = reshape_layer(bert_out[0])
            dropout_1 = dropout_layer_1(reshape_lambda)
            hidden_out = hidden_out(dropout_1)
            dropout_2 = dropout_layer_2(hidden_out)
            intent_out = intent_classifier(dropout_2)
            slot_out = slot_classifier(class_out)
            self.model = ks.models.Model(inputs = input_ids, outputs = [intent_out, slot_out])
            self.model.build(self.max_seq_len)
            
            
    
        
    def model_metrics(self, optimizer: str = "adam" , learning_rate: float = 1e-5, loss: str = "sparse" , metrics: list[str] = ["accuracy"]):
        if optimizer.lower() == "adam":
            optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            optimizer = ks.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = ks.optimizers.RMSprop(learning_rate=learning_rate)
            
        if loss.lower() == "categories":
            loss = ks.losses.CategoricalCrossentropy(from_logits = False)
        elif loss.lower() == "binary":
            loss = ks.losses.BinaryCrossentropy(from_logits = False)
        elif loss.lower() == "sparse":
            loss = ks.losses.SparseCategoricalCrossentropy(from_logits = False)
        
        for metric in metrics:
            if metric.lower() == "accuracy":
                metric = ks.metrics.Accuracy(name = "accuracy")
            elif metric.lower() == "precision":
                metric = ks.metrics.Precision(name = "precision")
            elif metric.lower() == "recall":
                metric = ks.metrics.Recall(name = "recall")
            elif metric.lower() == "f1":
                metric = ks.metrics.F1Score(name = "f1")
            elif metric.lower() == "mse":
                metrics = ks.metrics.MeanSquaredError(name = "mse")
            elif metric.lower() == "mae":
                metric = ks.metrics.MeanAbsoluteError(name = "mae")
            elif metric.lower() == "sparse":
                metrics = ks.metrics.SparseCategoricalAccuracy(name = "sparse")
        if not self.extended:
            self.optim = optimizer
            self.loss = loss
            self.metrics = metrics
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            self.optim = optimizer
            self.loss = loss
            self.metrics = metrics
            self.model.compile(
                optimizer = optimizer,
                loss={'intent_output': loss,
                      'slot_output': loss},
                metrics={'intent_out': metrics,
                         'slot_out': metrics})
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                   
    def model_summary(self):
        self.model.summary()
        
    def encode_input(self, data: np.ndarray, labels: np.ndarray):
        encoded_input = []
        encoded_output = []
        for utterance in tqdm(data):
            utterance = self.tokenizer.tokenize(utterance)
            utterance = ["<CLS>"] + utterance + ["<SEP>"]
            token_ids = self.tokenizer.convert_tokens_to_ids(utterance)
            if len(token_ids) < self.max_seq_len:
                padding = np.zeros((self.max_seq_len - len(token_ids),))
                token_ids = np.concatenate((token_ids, padding), axis=0)
            elif len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            encoded_input.append(token_ids)
        for label in labels:
            encoded_output.append(np.array(to_categorical(self.classes.index(label) , len(self.classes), dtype = "int32")))
            
        encoded_input = np.array(encoded_input)   
        encoded_output = np.array(encoded_output)
        # for output in encoded_output:
        #     output = np.array([output])
        #     print(output.shape)
        return encoded_input, encoded_output
            
    def train(self, train_data: pd.DataFrame, epochs: int = 10, batch_size: int = 32, verbose: int = 1, custom: bool = True):
        train_data = train_data.dropna()
        x_train = train_data['prompt'].to_numpy()
        y_train = train_data['intent'].to_numpy()
        x_train, y_train = self.encode_input(x_train, y_train)
        # y_train = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(0, 1))
        if not self.extended:
            if not custom:
                y_train = np.expand_dims(y_train, axis=1)
                self.model.fit(x_train, y_train.T, epochs = epochs, batch_size = batch_size, verbose = verbose)
            else:
                # y_train = np.expand_dims(y_train, axis=1)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
                for epoch in range(epochs):
                    print(f"\nStart of epoch {epoch}")
                    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset)):
                        with tf.GradientTape() as tape:
                            logits = self.model(x_batch_train, training=True)
                            loss_value = sparse_categorical_crossentropy(y_batch_train, logits)
                        grads = tape.gradient(loss_value, self.model.trainable_weights)
                        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

                        # Log every 100 batches.
                        if step % 100 == 0:
                            print(
                                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
                            )
                            print(f"Seen so far: {(step + 1) * batch_size} samples")    
        else:
            if not custom:
                y_train = np.expand_dims(y_train, axis=1)
                self.model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = verbose)
            else:
                # y_train = np.expand_dims(y_train, axis=1)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

                for epoch in range(epochs):
                    print(f"\nStart of epoch {epoch}")
                    # Separate losses for each output (assuming categorical labels)
                    total_intent_loss = 0
                    total_slot_loss = 0
                    num_batches = 0
                    for step, (x_batch_train, (y_intent_train, y_slot_train)) in tqdm(enumerate(train_dataset)):
                        with tf.GradientTape() as tape:
                            logits = self.model(x_batch_train, training=True)
                            intent_output, slot_output = logits  
                            intent_loss = self.loss(y_intent_train, intent_output)
                            slot_loss = self.loss(y_slot_train, slot_output)
                            loss_value = intent_loss + slot_loss 
                        grads = tape.gradient(loss_value, self.model.trainable_weights)
                        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
                        total_intent_loss += float(intent_loss)
                        total_slot_loss += float(slot_loss)
                        num_batches += 1
                        if step % 100 == 0:
                            avg_intent_loss = total_intent_loss / num_batches
                            avg_slot_loss = total_slot_loss / num_batches
                            print(f"Training loss (for 1 batch) at step {step}:")
                            print(f"  - Intent Loss: {avg_intent_loss:.4f}")
                            print(f"  - Slot Loss: {avg_slot_loss:.4f}")
                            print(f"Seen so far: {(step + 1) * batch_size} samples")
                            total_intent_loss = 0
                            total_slot_loss = 0
                            num_batches = 0
                        
    def predict(self, utterances: str):
        utterance_intent_pairs = []
        tokens = map(self.tokenizer.tokenize, utterances)
        tokens = map(lambda x: ["[CLS]"] + x + ["[SEP]"], tokens)
        token_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
        token_ids = map(lambda tids: tids + [0] * (self.max_seq_len - len(tids)), token_ids)
        token_ids = np.asarray(list(token_ids), dtype = "int32")
        predictions = self.model.predict(token_ids).argmax(axis= -1)
        for utterance, label in zip(utterances, predictions):
            utterance_intent_pairs.append((utterance, self.classes[label]))
        return utterance_intent_pairs
    
        
    def save_model(self, model_name: str):
        self.model_name = model_name
        self.model.save(self.model_name)
        
    def load_model(self, model_name: str):
        self.model_name = model_name
        self.model = ks.models.load_model(self.model_name)
        
    def evaluate(self, test_data: pd.DataFrame, batch_size: int = 32, verbose: int = 1):
        test_data = test_data.dropna()
        test_x, test_y = self.encode_input(test_data['text'].to_numpy(), test_data['intent'].to_numpy())
        y_pred = self.model.predict(test_x, batch_size = batch_size, verbose = verbose)
        return classification_report(test_y, y_pred)
    
    
        
def create_model_(model_name: str, max_seq_len: int, classes: list[str], model_path: str = None, dropout: float = 0.1, tokenizer: str = None, extended: bool = False, slot_classes: list[str] = None, multi: bool = False, optimizer: str = "adam" , learning_rate: float = 1e-5, loss: str = "sparse" , metrics: list[str] = ["accuracy"]) -> classifier:
    model = classifier(model_name, max_seq_len, classes, model_path, dropout, tokenizer, extended, slot_classes, multi)
    model.build()
    model.model_summary()
    model.model_metrics(optimizer, learning_rate, loss, metrics)
    return model


train = pd.read_csv("./Book1.csv")
print("train data: ", train)
classes = train.intent.unique().tolist()
print("possible intents: ", classes)
model = create_model_("bert", 38, classes, "bert-base-uncased", 0.1, "bert", False, [], False)
model.train(train, 20, 8, 1)
