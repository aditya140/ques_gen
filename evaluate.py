from utils import predict,sequence_to_text
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import io

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pickle
import dill
from datasets import load_question_dataset_test_iter
from test_serve import QuestionPredictor
from tqdm import tqdm


import hyperparams as hp
from decoding_helpers import Greedy, Teacher


def calc_bleu(ref,hyp):
    bleu={}
    bleu["bleu1"]=sentence_bleu(ref, hyp, weights=(1, 0, 0, 0))
    bleu["bleu2"]=sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0))
    bleu["bleu3"]=sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0))
    bleu["bleu4"]=sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25))
    #bleu={"bleu1":"NA","bleu2":"NA","bleu3":"NA","bleu4":"NA"}
    return bleu

def clean(sent):
    sent=sent.replace("<sos>","").replace("<eos>","").replace("<pad>","").replace("</sos>","").replace("</eos>","").replace("</pad>","").replace("?","")
    sent=sent.strip().split(" ")
    return sent

def evaluate_from_model():
    model=QuestionPredictor().model
    test_iter=load_question_dataset_test_iter(hp.dataset,hp.device)
    evaluate_metrics(model,test_iter)

def evaluate_metrics(model, test_iter):
    model.eval()
    fields = test_iter.dataset.fields
    greedy = Greedy(use_stop=True)
    df=pd.DataFrame(columns=["Answer","Target Question","Predited Question","bleu1","bleu2","bleu3","bleu4"])
    pbar=tqdm(test_iter, total=len(test_iter), unit=' batches',disable = hp.tqdm)
    for i, batch in enumerate(pbar):
        greedy.set_maxlen(len(batch.que[1:]))
        outputs, attention = model(batch.ans, greedy)
        seq_len, batch_size, vocab_size = outputs.size()

        preds = outputs.topk(1)[1]
        source = sequence_to_text(batch.ans[:, 0].data, fields['ans'])
        prediction = sequence_to_text(preds[:, 0].data, fields['que'])
        target = sequence_to_text(batch.que[1:, 0].data, fields['que'])
        bleu=calc_bleu([clean(target)],clean(prediction))
        df=df.append({"Answer":" ".join(clean(source)),"Target Question":" ".join(clean(target)),"Predited Question":" ".join(clean(prediction)),"bleu1":bleu["bleu1"],"bleu2":bleu["bleu2"],"bleu3":bleu["bleu3"],"bleu4":bleu["bleu4"]},ignore_index=True)
    df.to_csv("eval_results.csv")
    
    

        
        