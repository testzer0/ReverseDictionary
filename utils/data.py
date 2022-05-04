"""
Adithya Bhaskar, 2022.
This file, data.py, houses helper functions which help in loading, filtering and formatting data.
"""

import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import json
import os
import urllib.request
import zipfile
import pickle
import random
from collections import OrderedDict
from config import *
from utils.globals import *

def clean_text(text):
    """
    Cleans the text. For now, a no-op.
    """
    return text

def remove_duplicates(dictionary):
    """
    The EDMT dictionary has duplicates, e.g. two definitions of 'A'.
    Given a list of (word, meaning pairs), throws away all but the first definition for each word.
    https://stackoverflow.com/questions/29563953/most-pythonic-way-to-remove-tuples-from-a-list-if-first-element-is-a-duplicate
    """
    return list(OrderedDict(dictionary[::-1]).items())[::-1]

def split_definitions_webster(combined, min_word_count=3):
    """
    Split a string giving multiple definitions into its constitutents, and then filter by the minimum word count. Webster uses the convention -
    1. First definition 2. Second definition 3. ...
    """
    max_count = 0
    last_index = -1
    splits = [0]
    while True:
        search_for = "{}.".format(max_count+1)
        found_index = combined.find(search_for)
        if found_index <= last_index:
            break
        splits.append(found_index)
        last_index = found_index
        max_count += 1
    if max_count <= 1:
        defs = [combined.strip()]
    else:
        defs = [combined[i+2:j].strip() for i,j in zip(splits, splits[1:]+[None])]
    return [defn for defn in defs if len(defn.split()) >= min_word_count]

def split_definitions_edmt(combined, min_word_count=5):
    """
    Split a string giving multiple definitions into its constitutents, and then filter by the minimum word count. EDMT uses the convention -
    First definition ; Second definition ; ...
    """
    if ';' in combined:
        defs = [defn.strip() for defn in combined.split(';')]
    else:
        defs = [combined.strip()]
    return [defn for defn in defs if len(defn.split()) >= min_word_count]

def should_use_definition(word, definition, min_prefix_overlap=6, retain_probability = 0):
    """
    Some definitions are just poor for training. Consider:
    'The quality of being brutal' - brutalistic
    There are a lot of examples like this among Webster and EDMT -- we thus weed out those definitions where the word
    shares a prefix of length >= min_prefix_overlap with a word in the definition.
    We overlook a few cases with probability retain_probability.
    """
    min_prefix_overlap = min(min_prefix_overlap, len(word))
    ok = True
    for def_word in definition.split():
        if len(def_word) < min_prefix_overlap:
            continue
        if def_word[:min_prefix_overlap].lower() == word[:min_prefix_overlap].lower():
            ok = False
            break
    if ok:
        return True
    elif retain_probability > 0 and random.random() < retain_probability:
        return True
    else:
        return False

def process_dictionary(dictionary, name):
    """
    Split definitions and filter them.
    """
    processed = []
    for word, defn in dictionary:
        if name == 'webster':
            defs = split_definitions_webster(defn)
        elif name == 'edmt':
            defs = split_definitions_edmt(defn)
        else:
            defs = [defn]
        for split_defn in defs:
            if should_use_definition(word, split_defn):
                processed.append((word, split_defn))
    return processed

def glove_filter(dictionary):
    """
    Throw out those entries where the word is not in glove
    """
    dictionary = [(word, defn) for (word, defn) in dictionary if word in glove_vectors]
    return dictionary

def read_webster_dict(path="data/webster_dict.json"):
    with open(path) as f:
        webster = json.load(f)
    return remove_duplicates([(key.lower(), clean_text(value)) for key,value in webster.items()])

def read_edmt_dict(path="data/edmt_dict.json"):
    with open(path) as f:
        edmt = json.load(f)
    return remove_duplicates([(entry['word'].lower(), clean_text(entry['description'])) for entry in edmt])

def get_webster(log=False):
    """
    Download/read the Unabridged Webster's dictionary from the Gutenberg Project 2009.
    Also process it according to the flags in config.py.
    Optionally prints information about the dataset.
    """
    if dont_use_webster:
        return []
    
    if force_download or not os.path.isfile('data/webster_dict.json'):
        urllib.request.urlretrieve( \
            "https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/dictionary.json", \
            "data/webster_dict.json")
    webster = read_webster_dict()
    if log:
        print("Webster has {} word-definition pairs.".format(len(webster)))
        print(random.choice(webster))
    if process_dictionaries:
        webster = process_dictionary(webster, 'webster')
        if log:
            print("Webster has {} word-definition pairs after processing.".format(len(webster)))
            print(random.choice(webster))
    if filter_using_glove:
        webster = glove_filter(webster)
        if log:
            print("Webster has {} word-definition pairs after Glove filtering.".format(len(webster)))
            print(random.choice(webster))
    return webster

def get_edmt(log=False):
    """
    Download/read the EDMT dictionary.
    Also process it according to the flags in config.py.
    Optionally prints information about the dataset.
    """
    if dont_use_edmt:
        return []
    
    if force_download or not os.path.isfile('data/edmt_dict.json'):
        urllib.request.urlretrieve( \
            "https://raw.githubusercontent.com/eddydn/DictionaryDatabase/master/EDMTDictionary.json", \
            "data/edmt_dict.json")
    edmt = read_edmt_dict()
    if log:
        print("EDMT has {} word-definition pairs.".format(len(edmt)))
        print(random.choice(edmt))
    if process_dictionaries:
        edmt = process_dictionary(edmt, 'edmt')
        if log:
            print("EDMT has {} word-definition pairs after processing.".format(len(edmt)))
            print(random.choice(edmt))
    if filter_using_glove:
        edmt = glove_filter(edmt)
        if log:
            print("EDMT has {} word-definition pairs after Glove filtering.".format(len(edmt)))
            print(random.choice(edmt))
    return edmt

def get_wordnet(log=False):
    """
    Get (word, definition) pairs from Wordnet.
    """
    if dont_use_wordnet:
        return []
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    wordnet = [(synset.lemma_names()[0], synset.definition()) for synset in wn.all_synsets()]
    wordnet = [(word, defn) for (word, defn) in wordnet if '_' not in word]
    if log:
        print("WordNet has {} word-definition pairs.".format(len(wordnet)))
        print(random.choice(wordnet))
    if filter_using_glove:
        wordnet = glove_filter(wordnet)
        if log:
            print("WordNet has {} word-definition pairs after Glove filtering.".format(len(wordnet)))
            print(random.choice(wordnet))
    return wordnet

def get_unix(log=False):
    """
    Get (word, definition) pairs from the custom unix+vocabulary.com dataset.
    """
    unix = pickle.load(open('data/unix-dictionary.pkl', 'rb'))['dictionary']
    if log:
        print("Unix has {} word-definition pairs.".format(len(unix)))
        print(random.choice(unix))
    if filter_using_glove:
        unix = glove_filter(unix)
        if log:
            print("Unix has {} word-definition pairs after Glove filtering.".format(len(unix)))
            print(random.choice(unix))
    return unix

def get_data(log=False):
    """
    Just a wrapper to get all dictionaries. 
    If dont_use_X is set to False, get_X() will return an empty list.
    """
    return get_webster(log) + get_edmt(log) + get_wordnet(log) + get_unix(log)

if __name__ == '__main__':
    pass