"""
Adithya Bhaskar, 2022
Functions related to metrics.
"""

from config import *
from utils.globals import *
from utils.encdec import is_in_top_1_10_100
import numpy as np

def get_simple_baseline(inputs):
  gvec = np.zeros(100)
  n = 0
  for word in inputs.split():
    if word in glove_vectors:
      gvec += glove_vectors[word]
      n += 1
  if n > 0:
    gvec /= n
  return gvec

def eval_baseline():
  total = 0
  n1 = 0
  n10 = 0
  n100 = 0
  for sample in test_dataloader:
    test_defs = bert_tokenizer.batch_decode(sequences=sample[0], skip_special_tokens=True)
    targets = sample[2]
    for i in range(targets.shape[0]):
      word = torch.from_numpy(get_simple_baseline(test_defs[i])).to(device)
      top1, top10, top100 = is_in_top_1_10_100(targets[i], word)
      n1 += top1
      n10 += top10
      n100 += top100
      total += 1
      print(total, "done")
  print(total, n1, n10, n100)
  
def get_testing_accuracies(bert_lstm_model, dataloader, start=0, end=-1):
  total = 0
  n1 = 0
  n10 = 0
  n100 = 0
  for batch in dataloader:
    if start >= batch[2].shape[0]:
      start -= BATCH_SIZE
      continue
    batch_enc_def = batch[0].to(device)
    batch_attn_mask = batch[1].to(device)
    batch_targets = batch[2].to(device)
    outputs = bert_lstm_model(input_ids=batch_enc_def, attention_mask=batch_attn_mask)
    for i in range(start, outputs.shape[0]):
      actual = get_closest_word(batch_targets[i].cpu())
      top1, top10, top100 = is_in_top_1_10_100(actual, outputs[i])
      total += 1
      n1 += top1
      n10 += top10
      n100 += top100
      start = 0
      if i % 100 == 99:
        print("{} done.".format(i+1))
      if total == end:
        break
    if total == end:
      break
  p1 = 100.0 * n1 / total
  p10 = 100.0 * n10 / total
  p100 = 100.0 * n100 / total
  print("Top 1 accuracy  : 100% * {} / {} = {}%".format(n1, total, p1))
  print("Top 10 accuracy : 100% * {} / {} = {}%".format(n10, total, p10))
  print("Top 100 accuracy: 100% * {} / {} = {}%".format(n100, total, p100))
  return n1, n10, n100, total
  
def eval_model(bert_lstm_model, test_dataset):
    """
    Evaluate our model on the test set
    """
    print("If you want to run this from scratch, please write '0 0 0 0' into model/accuracy.txt")
    test_dataloader = get_dataloader(test_dataloader, test=True)
    while True:
        with open('model/accuracy.txt') as f:
            x = f.readlines()[0].split()
            x = [int(i) for i in x]
        if (x[0] == len(test_dataset)):
            break
        n1, n10, n100, total = get_testing_accuracies(bert_lstm_model, \
            test_dataloader, x[0], 64)
        x[0] += total
        x[1] += n1
        x[2] += n10
        x[3] += n100
        with open('model/accuracy.txt', 'w+') as f:
            f.write("{} {} {} {}\n".format(*x))
    print("Test Dataset Accuracy:")
    print("Top 1: {}%".format(100.0*x[1]/x[0]))
    print("Top 10: {}%".format(100.0*x[2]/x[0]))
    print("Top 100: {}%".format(100.0*x[3]/x[0]))
    return x