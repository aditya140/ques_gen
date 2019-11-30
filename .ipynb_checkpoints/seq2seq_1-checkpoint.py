#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import json
import glob


# In[21]:


def get_sentence(context,position,text):
    if "." in text[:-1]:
        return_2=True
    else:
        return_2=False
    context=context.split(".")
    count=0
    for sent in range(len(context)):
        if count+len(context[sent])>position:
            if return_2:
                return ".".join(context[sent:sent+2])
            else:
                return context[sent]
        else:
            count+=len(context[sent])+1
    return False

def create_dataset(data):
    load_failure=0
    try:
        if "data" in data.keys():
            data=data["data"]
    except:
        pass
    que_ans=[]
    for topic in data:
        for para in topic["paragraphs"]:
            for qa in para["qas"]:
                try:
                    res=[]
                    res.append(get_sentence(para["context"],qa["answers"][0]["answer_start"],qa["answers"][0]["text"]))
                    res.append(qa["question"])
                    que_ans.append(res)
                except:
                    load_failure+=1
    print("Load Failure : ",load_failure)
    return que_ans

def get_dataset():
    files=glob.glob("./data/*.json")
    data=[]
    for i in files:
        print("File name: ",i)
        with open(i,'rb') as f:
            data +=  create_dataset(json.load(f))
    return data


# In[22]:


def print_manipulation_():
    with open('./data/dev.json', 'r') as f:
        data = json.load(f)
    for topic in data:
        print("Topic :  " , topic["title"])
        print("="*5)
        for para in topic["paragraphs"]:
            print("Context: ",para["context"])
            for qa in para["qas"]:
                print("---"*10)
                print("\t Question: ",qa["question"])
                print("\t Answer: ",qa["answers"][0]["text"],"\t Answer start: ",qa["answers"][0]["answer_start"])
                print("\t Sent:",get_sentence(para["context"],qa["answers"][0]["answer_start"],qa["answers"][0]["text"]))
                print("\t Id: ",qa["id"])
                return
print_manipulation_()


# In[23]:


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device found : ", device)


# In[24]:


def print_sample_dataset(que_ans):
    print("============Sample dataset============\n")
    for i in range(len(que_ans)):
        print("Answer Sentence: \t",que_ans[i][0])
        print("Question: \t",que_ans[i][1])
        print("=="*5)
print_sample_dataset(get_dataset()[:5])


# In[25]:


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            


# In[26]:




# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def prepare_data(inp_lang=None,opt_lang=None):
    if inp_lang==None:
        inp_lang=Lang("input_answers")
    if opt_lang==None:
        opt_lang=Lang("output_questions")
    que_ans=get_dataset()
    
    que_ans_n=[[normalizeString(i[0]),normalizeString(i[1])] for i in que_ans]
    que_ans_norm=[]
    for i in range(len(que_ans_n)):
        if len(que_ans_n[i][0].split(" "))<100 and len(que_ans_n[i][1].split(" "))<100:
inp_lang.addSentence(que_ans_n[i][0])
opt_lang.addSentence(que_ans_n[i][1])
que_ans_norm.append([que_ans_n[i][0],que_ans_n[i][1]])            
    return inp_lang,opt_lang,que_ans_norm


inp_lang,opt_lang,que_ans_norm=prepare_data()
print("__"*10)
print("question: \t",(que_ans_norm[10][1]))
print("answer: \t",(que_ans_norm[10][0]))


print("question: \t",
      [opt_lang.word2index[word] for word in que_ans_norm[10][1].split(' ')])
print("answer: \t",
      [inp_lang.word2index[word] for word in que_ans_norm[10][0].split(' ')])

print("Dataset size: ",len(que_ans_norm))


# In[27]:


data=pd.DataFrame(que_ans_norm,columns=["ans","que"])
data=data.drop_duplicates()
print("Non Duplicates Dataset size: ",data.shape)


# In[28]:


import numpy as np
data["ans_len"]=data["ans"].apply(lambda x: len(x.split(" ")))
data["que_len"]=data["que"].apply(lambda x: len(x.split(" ")))

(data.head())
print("Max Question lenght: ", max(data["que_len"].values), "\tAverage Question lenght: ", np.mean(data["que_len"].values))
print("Max Answer lenght: ", max(data["ans_len"].values), "\tAverage Answer lenght: ", np.mean(data["ans_len"].values))
que_ans_norm=data[["ans","que"]].values


# In[29]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.GRU(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[30]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.GRU = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.GRU(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[31]:



MAX_LENGTH=100
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.GRU = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.GRU(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[32]:


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(inp_lang, pair[0])
    target_tensor = tensorFromSentence(opt_lang, pair[1])
    return (input_tensor, target_tensor)


# In[33]:


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[34]:


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[35]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(que_ans_norm))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# In[36]:


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


# In[37]:


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(inp_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(opt_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# In[38]:


hidden_size = 500
encoder1 = EncoderRNN(inp_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, opt_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 80000, print_every=1000)


# In[39]:


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(que_ans_norm)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[55]:


evaluateRandomly(encoder1, attn_decoder1)


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
output_words, attentions = evaluate(
    encoder1, attn_decoder1, "in provost susan hockfield became the president of the massachusetts institute of technology")
plt.matshow(attentions.numpy())


# In[51]:


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

evaluateAndShowAttention("in provost susan hockfield became the president of the massachusetts institute of technology")


# In[ ]:


torch.save(encoder1.state_dict(), 'encoder-1.dict')
torch.save(attn_decoder1.state_dict(), 'decoder-1.dict')


# In[47]:


import pickle
with open("inp_lang-1.p",'wb') as f:
    pickle.dump(inp_lang,f)
    
with open("opt_lang-1.p",'wb') as f:
    pickle.dump(opt_lang,f)


# In[48]:


torch.save(encoder1.state_dict(), 'encoder-1.dict')
torch.save(attn_decoder1.state_dict(), 'decoder-1.dict')


# In[56]:


print(1)


# In[ ]:




