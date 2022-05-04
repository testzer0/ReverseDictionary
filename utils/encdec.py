"""
This file exports functions for encoding and decoding data. It also has 
functions to get the top-k words.
"""

from config import *
from utils.globals import *
from sklearn.model_selection import train_test_split
import torch
import urllib.request
import zipfile
import pickle
import math
import numpy as np

def load_glove_vectors(log=False):
    """
    Loads glove_vectors.
    """
    global glove_vectors
    if force_download or not os.path.isfile('data/glove_6B_100d.pkl'):
        urllib.request.urlretrieve ("http://nlp.stanford.edu/data/glove.6B.zip", "data/glove.6B.zip")
        with zipfile.ZipFile("data/glove.6B.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
        if log:
            print("Indexing word vectors.")
        embeddings_index = {}
        f = open("data/glove.6B.100d.txt", encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        if log:
            print('Found %s word vectors.' % len(embeddings_index))
        pickle.dump({'embeddings_index' : embeddings_index } , open('data/glove_6B_100d.pkl', 'wb'))
    glove_vectors = pickle.load(open('data/glove_6B_100d.pkl', 'rb'))['embeddings_index']
    if log:
        print("Example glove vector for 'sprint': ", glove_vectors['sprint'])

def encode_data(data, tokenizer, max_def_length=128):
    """
    Removes newlines, then encodes the definition of each word in the input. Prunes away those inputs whose encoded length exceeds
    max_length. Also returns the encoded gold-truth outputs.
    Observation - the encoded word is anywhere from 2 to 9 tokens long. Since a word may correspond to more than one token,
    it is hard to enforce a 1-token rule. Hence, we just enforce that the output is at most 10 tokens long.
    """
    num_total = len(data)
    encoded_def = []
    encoded_def_attn_masks = []
    encoded_targets = []
    for i in range(num_total):
        word = data[i][0]
        definition = data[i][1].replace('\n','')
        iids = tokenizer.encode(definition, add_special_tokens=True, padding="max_length", max_length=128, return_tensors="pt")[0]
        if iids.shape[-1] != 128:
            continue
        attn_mask = (iids != tokenizer.pad_token_id).int()
        if word == '':
            target = torch.zeros(100)
        else:
            target = torch.tensor(glove_vectors[word])
        encoded_def.append(iids)
        encoded_def_attn_masks.append(attn_mask)
        encoded_targets.append(target)
    encoded_def = torch.stack(encoded_def)
    encoded_def_attn_masks = torch.stack(encoded_def_attn_masks)
    encoded_targets = torch.stack(encoded_targets)
    return (encoded_def, encoded_def_attn_masks, encoded_targets)

def train_val_test_split(encoded_dataset):
    """
    Splits the dataset into train, validation and test datasets. Currently, 92%, 4.8% and 3.2% of the samples go to the training, validation
    and test sets, respectively.
    """
    train_enc_def, val_test_enc_def, train_targets, val_test_targets = train_test_split(encoded_dataset[0], encoded_dataset[2], random_state=199, test_size=0.08)
    train_attn_masks, val_test_attn_masks, _, _ = train_test_split(encoded_dataset[1], encoded_dataset[2], random_state=199, test_size=0.08)
    val_enc_def, test_enc_def, val_targets, test_targets = train_test_split(val_test_enc_def, val_test_targets, random_state=1700, test_size=0.4)
    val_attn_masks, test_attn_masks, _, _ = train_test_split(val_test_attn_masks, val_test_targets, random_state=1700, test_size=0.4)

    return {
        'train' : (train_enc_def, train_attn_masks, train_targets),
        'validation' : (val_enc_def, val_attn_masks, val_targets),
        'test' : (test_enc_def, test_attn_masks, test_targets)
    }

def get_encoded_and_split_dataset(dataset):
    """
    Simple wrapper to get the processed dataset.
    """
    return train_val_test_split(encode_data(dataset, bert_tokenizer))

def has_blocked_chars(word):
    """
    Prune away words with spurious characters such as @
    """
    return (word in sample_bad_words) or any(not char.isalpha() for char in word)

def get_k_closest_words(vec, k=5, skip_implausible=True):
    """
    Returns top k closest words when comparing - for now only k=1 is supported.
    """
    vec = vec.detach().cpu().numpy().flatten()
    closest = [None] * k
    distances = [math.inf] * k
    if glove_vectors is None:
        load_glove_vectors()
    for word, wvec in glove_vectors.items():
        if skip_implausible and has_blocked_chars(word):
            continue
        distance = np.linalg.norm(wvec-vec)
        ind = 0
        while ind < k and distances[ind] < distance:
            ind += 1
        if ind < k:
            closest = closest[:ind] + [word] + closest[ind:-1]
            distances = distances[:ind] + [distance] + distances[ind:-1]
    return closest

def get_closest_word(vec, skip_implausible=True):
    """
    Gets the closest word among the glove words to the given vector
    """
    vec = vec.detach().cpu().numpy().flatten()
    closest = None
    dmin = math.inf
    if glove_vectors is None:
        load_glove_vectors()
    for word, wvec in glove_vectors.items():
        if skip_implausible and has_blocked_chars(word):
            continue
        distance = np.linalg.norm(wvec-vec)
        if distance < dmin:
            closest = word
            dmin = distance
    return closest

def is_in_top_1_10_100(word, vec):
    """
    Returns three booleans depicting whether the word is among the top 1, 10, and 100
    closest ones respectively in terms of word vector distance to vec.
    """
    words_100 = get_k_closest_words(vec=vec, k=100)
    if word not in words_100:
        return (0,0,0)
    else:
        idx = words_100.index(word)
        if idx == 0:
            return (1,1,1)
        elif idx < 10:
            return (0,1,1)
        else:
            return (0,0,1)
        
if __name__ == '__main__':
    pass