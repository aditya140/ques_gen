import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
import pickle
from datetime import date
import dill
from beam import BeamHelper

import hyperparams as hp
from decoding_helpers import Greedy, Teacher

def sequence_to_text(sequence, field):
    return " ".join([field.vocab.itos[int(i)] for i in sequence])


def text_to_sequence(text, field):
    return [field.vocab.stoi[word] for word in text]

def predict_beam(model,sent,fields,beam_size):
    source=([fields['ans'].vocab.stoi[fields['ans'].init_token]]+text_to_sequence(fields['ans'].preprocess(sent),fields['ans'])+[fields['ans'].vocab.stoi[fields['ans'].eos_token]])
    source=[[i] for i in source]
    source=torch.LongTensor(source)

    model.eval()
    total_loss = 0
    beam = BeamHelper(beam_size=beam_size, maxlen=hp.max_len)
    source=source.to(hp.device)
    best_score, best_seq = model(source, beam)
    #preds = outputs.topk(1)[1]
    prediction = sequence_to_text(best_seq, fields['que'])
    return prediction


def predict(model, sent, fields):
    model.eval()
    total_loss = 0
    greedy = Greedy()
    source=[text_to_sequence(sent, fields["ans"])]
    source = torch.LongTensor(source)
    source=source.to(hp.device)
    greedy.set_maxlen(hp.max_len)
    outputs, attention = model(source, greedy)
    
    seq_len, batch_size, vocab_size = outputs.size()

    preds = outputs.topk(1)[1]
    prediction = sequence_to_text(preds[:, 0].data, fields['que'])
    #attention_plot = show_attention(attention[0],prediction, sent, return_array=True)

    return prediction
    


def evaluate(model, val_iter, writer, step):
    model.eval()
    total_loss = 0
    fields = val_iter.dataset.fields
    greedy = Greedy()
    random_batch = np.random.randint(0, len(val_iter) - 1)
    for i, batch in enumerate(val_iter):
        greedy.set_maxlen(len(batch.que[1:]))
        outputs, attention = model(batch.ans, greedy)
        seq_len, batch_size, vocab_size = outputs.size()
        loss = F.cross_entropy(outputs.view(seq_len * batch_size, vocab_size),
                               batch.que[1:].view(-1),
                               ignore_index=hp.pad_idx)
        total_loss += loss.item()

        # tensorboard logging
        if i == random_batch:
            preds = outputs.topk(1)[1]
            source = sequence_to_text(batch.ans[:, 0].data, fields['ans'])
            prediction = sequence_to_text(preds[:, 0].data, fields['que'])
            target = sequence_to_text(batch.que[1:, 0].data, fields['que'])
            attention_plot = show_attention(attention[0],
                                            prediction, source, return_array=True)

            #writer.add_figure('Attention', attention_plot, step)
            writer.add_text('Source: ', source, step)
            writer.add_text('Prediction: ', prediction, step)
            writer.add_text('Target: ', target, step)
    writer.add_scalar('val_loss', total_loss / len(val_iter), step)


def train(model, optimizer, scheduler, train_iter, val_iter,
          num_epochs, teacher_forcing_ratio=0.5, step=0):
    model.train()
    writer = SummaryWriter()
    teacher = Teacher(teacher_forcing_ratio)
    fields = val_iter.dataset.fields
    d1 = date.today().strftime("%d-%m-%Y")
    run_version=np.random.randint(low=10000, high=99999)
    
    with open(f'models/fields/fields_{d1}_{run_version}.pkl', 'wb')as f:
         dill.dump(fields,f)
    
    #pickle not working
#     with open(f'models/fields/fields_{d1}_{run_version}.pkl', 'wb') as output:
#         pickle.dump(fields,output)
    for _ in tqdm(range(num_epochs), total=num_epochs, unit=' epochs',disable = hp.tqdm):
        pbar = tqdm(train_iter, total=len(train_iter), unit=' batches',disable = hp.tqdm)
        for b, batch in enumerate(pbar):
            optimizer.zero_grad()
            teacher.set_targets(batch.que)
            outputs, masks = model(batch.ans, teacher)
            seq_len, batch_size, vocab_size = outputs.size()
            loss = F.cross_entropy(outputs.view(seq_len * batch_size, vocab_size),
                                   batch.que[1:].view(-1),
                                   ignore_index=hp.pad_idx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2)  # prevent exploding grads
            scheduler.step()
            optimizer.step()

            # tensorboard logging
            pbar.set_description(f'loss: {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('lr', scheduler.lr, step)
            step += 1
            
        torch.save(model.state_dict(), f'checkpoints/seq2seq_{step}.pt')
        torch.save(model, f'models/model/seq2seq_{d1}_{run_version}.pt')
        evaluate(model, val_iter, writer, step)
        model.train()

def show_attention(attention, prediction=None, source=None, return_array=False):
    plt.figure(figsize=(14, 6))
    sns.heatmap(attention,
                xticklabels=prediction.split(),
                yticklabels=source.split(),
                linewidths=.05,
                cmap="Blues")
    plt.ylabel('Source (German)')
    plt.xlabel('Prediction (English)')
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    return plt
#     if return_array:
#         plt.tight_layout()
#         buff = io.BytesIO()
#         plt.savefig(buff, format='png')
#         buff.seek(0)
#         return np.array(Image.open(buff))
