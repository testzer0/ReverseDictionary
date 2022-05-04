"""
Adithya Bhaskar, 2022.
The main file that takes input and drives the other modules.
"""

from config import *
from utils.globals import *
from utils.data import get_data
from utils.encdec import get_encoded_and_split_dataset, \
    get_k_closest_words, get_closest_word, encode_data
from model.model import get_model_optimizer_and_scheduler, \
    load_latest_checkpt
from model.metrics import eval_model
import argparse

dataset = None
bert_lstm_model = None

def get_word_from_single_def(definition, tokenizer, bert_lstm_model):
    """
    Given a definition, returns the word with the highest model-assigned score.
    """
    single_word_dset = [["", definition]]
    encoded = encode_data(single_word_dset, tokenizer)
    defn = encoded[0].to(device)
    mask = encoded[1].to(device)
    outputs = bert_lstm_model(input_ids=defn, attention_mask=mask)
    return [get_closest_word(outputs[0])]

def get_k_closest_words_from_single_def(definition, tokenizer, bert_lstm_model):
    """
    Given a definition, returns the k words with the highest model-assigned score.
    """
    single_word_dset = [["", definition]]
    encoded = encode_data(single_word_dset, tokenizer)
    defn = encoded[0].to(device)
    mask = encoded[1].to(device)
    outputs = bert_lstm_model(input_ids=defn, attention_mask=mask)
    return get_k_closest_words(outputs[0], k=5)

def train():
    """
    Train the model.
    """
    global dataset, bert_lstm_model
    data = get_data()
    dataset = get_encoded_and_split_dataset(data)
    train_dataloader = get_dataloader(dataset['train'])
    validation_dataloader = get_dataloader(dataset['validation'])
    bert_lstm_model, optimizer, scheduler = \
        get_model_optimizer_and_scheduler(train_dataloader)
    train_bert_lstm(bert_lstm_model, optimizer, scheduler, \
        train_dataloader, validation_dataloader)
    print("Training done.")
    
def evaluate():
    """
    Evaluate the model on the test set.
    @trained - Whether train() was called before this
    """
    global dataset, bert_lstm_model
    if (dataset is None) or (bert_lstm_model is None):
        # train() was not called before this
        data = get_data()
        dataset = get_encoded_and_split_dataset(data)
        
        # Passing None returns only the model
        bert_lstm_model = get_model_optimizer_and_scheduler(None)
        load_latest_checkpt(bert_lstm_model)
    bert_lstm_model.eval()
    eval_model(bert_lstm_model, dataset['test'])
    
def get_word(definition):
    """
    Get a word from the definition
    """
    global bert_lstm_model
    if (bert_lstm_model is None):
        # Passing None returns only the model
        bert_lstm_model = get_model_optimizer_and_scheduler(None)
        load_latest_checkpt(bert_lstm_model)
    bert_lstm_model.eval()
    return get_word_from_single_def(definition, bert_tokenizer, bert_lstm_model)[0]

def get_k_words(definition):
    """
    Get k closest words from the definition
    """
    global bert_lstm_model
    if (bert_lstm_model is None):
        # Passing None returns only the model
        bert_lstm_model = get_model_optimizer_and_scheduler(None)
        load_latest_checkpt(bert_lstm_model)
    bert_lstm_model.eval()
    return get_k_closest_words_from_single_def(definition, bert_tokenizer, bert_lstm_model)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model", action="store_true")
    parser.add_argument("--eval", help="Evaluate the model", action="store_true")
    parser.add_argument("--defn", help="The definition", type=str)
    parser.add_argument("--file", help="File containing definition", type=str)
    parser.add_argument("--top", help="Print top k words -- 1 by default", type=int)
    args = parser.parse_args()
    did = False
    if args.train:
        did = True
        train()
    if args.eval:
        did = True
        evaluate()
    defn = ""
    if args.file is not None:
        did = True
        if args.defn is not None:
            print("Both definition and file passed!")
            exit(-1)
        try:
            defn = open(args.file, "r").readlines[0].strip()
        except:
            print("An exception occurred when trying to read file!")
            exit(-1)
    elif args.defn:
        did = True
        defn = args.defn
    n = 1
    if args.top is not None:
        did = True
        n = args.top
        if (n <= 0):
            print("Top value is <= 0!")
            exit(-1)
    if not did:
        parser.print_help()
        parser.exit()
    elif not args.defn:
        exit(0)
    elif n == 1:
        print("Predicted word: {}".format(get_word(defn)))
    else:
        print("Top {} predicted words:".format(n))
        words = get_k_words(defn)
        for i in range(n):
            print("{}. {}".format(i+1, words[i]))