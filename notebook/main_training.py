import pickle

from build_vocab import *
from build_answers import *

# train2014
answers_path = '/home/ubuntu/dotCuda/notebook/answers.pkl'
with open(answers_path, 'rb') as f:
    answers = pickle.load(f)

vocab_path = '../../dotCuda/notebook/vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
    
# val2014
answers_path = '/home/ubuntu/dotCuda/notebook/valanswers.pkl'
with open(answers_path, 'rb') as f:
    valanswers = pickle.load(f)

vocab_path = '../../dotCuda/notebook/valvocab.pkl'
with open(vocab_path, 'rb') as f:
    valvocab = pickle.load(f)
    
    
from dataset import *

train_dataset = COCODataset(vocab=vocab, answers=answers)
val_dataset = COCODataset(vocab=valvocab, answers=valanswers)


# import network 
from network_v5 import *
import torch
device = torch.device('cuda')

vocab_size = len(vocab)
model = ConcatNet(vocab_size).to(device)


# import trainer
from trainer import VQATrainer

trainer = VQATrainer(model, device)

trained_model, statistics = trainer.train(train_dataset, val_dataset, collate_fn=collate, e_break=10000)

with open('stats.st', 'wb') as statfile:
    pickle.dump(statistics, statfile)