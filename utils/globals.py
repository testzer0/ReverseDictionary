"""
Adithya Bhaskar, 2022.
This file lists some global variables that are used by many modules.
It this makes sense to put them in their own file.
"""

from config import *
from transformers import AutoTokenizer
import urllib.request
import zipfile
import os
import pickle
import torch

def get_device():
    """
    Set device and return it, the first time.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def load_glove_vectors(log=False):
    """
    Loads glove_vectors.
    """
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
    return glove_vectors

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
glove_vectors = load_glove_vectors(False)
device = get_device()
        
if __name__ == '__main__':
    pass