"""
This file contains functions to create and train the model.
"""

from config import *
from utils.globals import *
import os
import re
import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW

import transformers
from transformers import AutoTokenizer, BertForMaskedLM, BertModel, get_linear_schedule_with_warmup


NUM_STEPS = 0

class BertLSTM(nn.Module):
  """
  BERT -> LSTM -> Linear
  """
  def __init__(self, out_dim=100, seq_len=128):
    super().__init__()
    self.out_dim = out_dim
    self.seq_len = seq_len
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.hidden_size = self.bert.config.hidden_size
    self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)
    self.Linear = nn.Linear(self.hidden_size*2, self.out_dim)
    self.train_mode = True

  def train(self):
    self.train_mode = True

  def eval(self):
    self.train_mode = False

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids,attention_mask)
    encoded_layers, pooled_output = outputs.last_hidden_state, outputs.pooler_output
    seq_lens = encoded_layers.shape[0] * [self.seq_len]
    encoded_layers = encoded_layers.permute(1, 0, 2)
    enc_hiddens, (last_hidden, last_cell) = self.LSTM(nn.utils.rnn.pack_padded_sequence(encoded_layers, seq_lens))
    output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    output_hidden = nn.functional.dropout(output_hidden,0.2)
    if self.train_mode:
      output_hidden = nn.functional.dropout(output_hidden,0.2)
    return self.Linear(output_hidden)

class BertMultiLSTM(nn.Module):
  """
  BERT -> 4 x (LSTM + Dropout) -> Linear
  """
  def __init__(self, out_dim=100, seq_len=128):
    super().__init__()
    self.out_dim = out_dim
    self.seq_len = seq_len
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.hidden_size = self.bert.config.hidden_size
    self.LSTM = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=4, dropout=0.1, bidirectional=True)
    self.Linear = nn.Linear(self.hidden_size*2, self.out_dim)
    self.train_mode = True

  def train(self):
    self.train_mode = True

  def eval(self):
    self.train_mode = False
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids,attention_mask)
    encoded_layers, pooled_output = outputs.last_hidden_state, outputs.pooler_output
    seq_lens = encoded_layers.shape[0] * [self.seq_len]
    encoded_layers = encoded_layers.permute(1, 0, 2)
    enc_hiddens, (last_hidden, last_cell) = self.LSTM(nn.utils.rnn.pack_padded_sequence(encoded_layers, seq_lens))
    output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
    if self.train_mode:
      output_hidden = nn.functional.dropout(output_hidden,0.2)
    return self.Linear(output_hidden)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / labels_flat.shape[0]

def get_max_checkpt(checkpt_dir):
    max_checkpt = 0
    for filename in os.listdir(checkpt_dir):
        if re.match(r"checkpt-([0-9]+).pt", filename):
            checkpt_num = int(filename.split('.')[-2].split('-')[-1])
        if checkpt_num > max_checkpt:
            max_checkpt = checkpt_num
    return max_checkpt

def load_latest_checkpt(bert_lstm_model, checkpt_dir=CHECKPT_DIR):
    if force_restart_training:
        return 0
    mx_checkpt = get_max_checkpt(checkpt_dir)
    if mx_checkpt > 0:
        checkpt_file = os.path.join(checkpt_dir, "checkpt-{}.pt".format(mx_checkpt))
        bert_lstm_model.load_state_dict(torch.load(checkpt_file, map_location=device))
    return mx_checkpt

def get_dataloader(dataset, test=False):
    """
    Returns a dataloader for the dataset.
    Uses a Random Sampler for train and validation, and Sequential sampler for test.
    """
    dataset = TensorDataset(*dataset)
    if test:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)
    return dataloader

def get_model_optimizer_and_scheduler(train_dataloader=None, log=False, ):
    """
    Get the model, optimizer and scheduler.
    If train_dataloader is None, it is assumed that only the model is requested (e.g. for eval).
    Hence in this case the optimizer and scheduler are not returned.
    """
    global NUM_STEPS
    if use_multi_layers:
        bert_lstm_model = BertMultiLSTM()
    else:
        bert_lstm_model = BertLSTM()
    if log:
        print(bert_lstm_model)
    if torch.cuda.is_available():
        print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
        bert_lstm_model.cuda()
    else:
        print("No GPUs available, using CPU")
    if train_dataloader is None:
        return bert_lstm_model
    NUM_STEPS = len(train_dataloader) * NUM_TARGET_EPOCHS
    optimizer = AdamW(bert_lstm_model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=NUM_STEPS)
    return bert_lstm_model, optimizer, scheduler

def train_bert_lstm(bert_lstm_model, optimizer, scheduler, \
    train_dataloader, validation_dataloader,):
    loss_values = []
    start_epoch = load_latest_checkpt(bert_lstm_model) # 0-indexed
    scheduler.last_epoch = start_epoch - 1
    bert_lstm_model.train()
    for epoch in range(start_epoch, NUM_EPOCHS):
        print("Using BERT-LSTM model")
        print("======== Epoch {} / {} ========".format(epoch+1, NUM_EPOCHS))
        print("Training phase")
        epoch_start = time.time()
        epoch_loss = 0
        bert_lstm_model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - epoch_start)
                print("Batch {} of {}. Elapsed {}".format( \
                    step, len(train_dataloader), elapsed))
            batch_enc_def = batch[0].to(device)
            batch_attn_mask = batch[1].to(device)
            batch_targets = batch[2].to(device) # These are the glove vectors
            bert_lstm_model.zero_grad()
            outputs = bert_lstm_model(input_ids=batch_enc_def, \
                attention_mask=batch_attn_mask)
            # This function takes logits and labels
            MSE = nn.MSELoss(reduction='none')
            loss = MSE(outputs, batch_targets)
            loss = torch.mean(torch.sum(loss, axis=1))
            epoch_loss += loss
            loss.backward()
            clip_grad_norm_(bert_lstm_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = epoch_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("Average training loss for epoch {} : {}".format(epoch+1, avg_train_loss))
        print("Epoch took {}".format(format_time(time.time()-epoch_start)))

        print("\nValidation phase")
        val_start = time.time()
        bert_lstm_model.eval()
        val_loss, val_accuracy = 0, 0
        batch_eval_steps, batch_eval_examples = 0, 0
        for batch in validation_dataloader:
            batch = tuple(tup.to(device) for tup in batch)
            batch_enc_def, batch_attn_mask, batch_targets = batch
            with torch.no_grad():
                outputs = bert_lstm_model(input_ids=batch_enc_def, \
                    attention_mask=batch_attn_mask)
            MSE = nn.MSELoss(reduction='none')
            loss = MSE(outputs, batch_targets)
            loss = torch.mean(torch.sum(loss, axis=1))
            val_loss += loss
        avg_val_loss = val_loss / len(validation_dataloader)
        print("Validation loss: {}".format(avg_val_loss))
        print("Validation took {}".format(format_time(time.time()-val_start)))
        if save:
            checkpt_path = os.path.join(CHECKPT_DIR, "checkpt-{}.pt".format(epoch+1))
            torch.save(bert_lstm_model.state_dict(), checkpt_path)

if __name__ == '__main__':
    pass