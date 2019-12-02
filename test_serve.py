import hyperparams as hp
import re
import glob
import os
import pickle
import utils
import dill
import torch
import unicodedata


def unicodeToAscii(s):
        return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    
def _normalize(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    return s

def load_model(latest=True,name=None):
    if name==None:
        if latest:
            field_paths = glob.glob("./models/fields/*")
            latest_file = max(field_paths, key=os.path.getctime)
            pattern=re.compile("fields_([0-9\-]*)_(\d*).pkl")
            
            with open(latest_file,'rb')as f:
                fields=dill.load(f)
            #with open(f"./models/fields/{latest_file}",'rb') as handle:
             #   fields=pickle.load(handle)
            date_version=re.findall(pattern,latest_file)
            model_path=f"./models/model/seq2seq_{date_version[0][0]}_{date_version[0][1]}.pt"
            model = torch.load(model_path)
            return model,fields
        print("Please specify correct arguments")
    else:
        pattern=re.compile("seq2seq_([0-9\-]*)_(\d*).pt")
        date_version=re.findall(pattern,name)
        field_path=f"./models/fields/fields_{date_version[0][0]}_{date_version[0][1]}.pkl"
        with open(field_path,'rb')as f:
            fields=dill.load(f)
        #with open(field_path,'rb') as handle:
            #fields=pickle.load(handle)
        model_path=f"./models/model/seq2seq_{date_version[0][0]}_{date_version[0][1]}.pt"
        model = torch.load(model_path)
        return model,fields
        
class QuestionPredictor(object):
    def __init__(self,model=None,latest=True):
        self.model,self.fields=load_model(latest=latest,name=model)
        self.model.to(hp.device)
    def predict(self,sent,beam=False,beam_size=3):
        if beam:
            return utils.predict_beam(model=self.model,sent=_normalize(sent),fields=self.fields,beam_size=beam_size)
        return utils.predict(model=self.model,sent=_normalize(sent),fields=self.fields)
