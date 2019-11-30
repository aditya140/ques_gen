# from torchtext test cases
import spacy
from torchtext.data import Field, BucketIterator,TabularDataset
import glob
import json
import pandas as pd
import unicodedata
import re
from sklearn.model_selection import train_test_split
import hyperparams as hp

class QGenDataset(object):
    def __init__(self,dataset):
        self.ans=[]
        self.que=[]
        self.ans_que=[]

        self.ans_train=[]
        self.que_train=[]
        self.ans_que_train=[]

        self.ans_test=[]
        self.que_test=[]
        self.ans_que_test=[]

        self.ans_val=[]
        self.que_val=[]
        self.ans_que_val=[]
        self.dataset=dataset
        if "marco" in dataset:
            self._create_marco_data(normalize=True)
        if "squad" in dataset:
            self._fetch_data_squad(normalize=True)
        self.to_csv()
        
    
    def _create_marco_data(self,normalize=True):
        marco_path="./data/marco/*"
        for file in glob.glob(marco_path):
            with open(file,'r') as f:
                data=json.load(f)
            for i in data["passages"].keys():
                if data["answers"][i]!=["No Answer Present."]:
                    self.ans.append(self._normalize(".".join([x["passage_text"] for x in data["passages"][i] if x["is_selected"]==1])))
                    self.que.append(self._normalize(data["query"][i]))
        
        


    def _fetch_data_squad(self,normalize=True):
        self.ans_que=self._get_dataset(normalize=normalize)
        data=pd.DataFrame(self.ans_que,columns=["ans","que"])
        data=data.drop_duplicates()
        self.ans_que=data[["ans","que"]].values
        for i in self.ans_que:
            self.ans.append(i[0])
            self.que.append(i[1])

        
    def to_csv(self):
        raw_data = {'ans' : [line for line in self.ans], 'que': [line for line in self.que]}
        df = pd.DataFrame(raw_data, columns=["ans", "que"])
        # remove very long sentences and sentences where translations are 
        # not of roughly equal length
        df['ans_len'] = df['ans'].astype(str).str.count(' ')
        df['que_len'] = df['que'].astype(str).str.count(' ')
        df = df.query('ans_len < 80 & que_len < 80')
        df = df.drop_duplicates()
        print("Dataframe size:",df.shape)
        train, TEST = train_test_split(df, test_size=0.3)
        test, val = train_test_split(TEST, test_size=0.5)
        train.to_csv("./.data/train.csv", index=False)
        val.to_csv("./.data/val.csv", index=False)
        test.to_csv("./.data/test.csv", index=False)

    @staticmethod
    def _get_sentence(context,position,text):
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

    def _create_dataset(self,data,normalize=True):
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
                        if normalize:
                            res.append(self._normalize(self._get_sentence(para["context"],qa["answers"][0]["answer_start"],\
                                                                          qa["answers"][0]["text"])))
                            res.append(self._normalize(qa["question"]))
                        else:
                            res.append(self._get_sentence(para["context"],qa["answers"][0]["answer_start"],\
                                                          qa["answers"][0]["text"]))
                            res.append(qa["question"])
                        que_ans.append(res)
                    except:
                        load_failure+=1
        print("Load Failure : ",load_failure)
        return que_ans

    def _get_dataset(self,normalize=True):
        files=glob.glob("./data/*.json")
        data=[]
        for i in files:
            with open(i,'rb') as f:
                data +=  self._create_dataset(json.load(f),normalize=normalize)
        return data

    @staticmethod
    def unicodeToAscii(s):
        return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    def _normalize(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s





def load_question_dataset(batch_size,dataset,device=0):
    spacy_en = spacy.load('en')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    inp_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')
    opt_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

    dataset=QGenDataset(dataset)
    
    # associate the text in the 'English' column with the EN_TEXT field, # and 'French' with FR_TEXT
    data_fields = [('ans', inp_lang), ('que', opt_lang)]
    train,val,test= TabularDataset.splits(path='./.data/', train='train.csv', validation='val.csv',test= "test.csv", format='csv', fields=data_fields)

    inp_lang.build_vocab(train,val,test)
    opt_lang.build_vocab(train,val,test)

    train_iter = BucketIterator(train, batch_size=batch_size, \
            device=device, repeat=False , sort_key=lambda x: len(x.que), shuffle=True)
    val_iter = BucketIterator(val, batch_size=batch_size, \
            device=device, sort_key=lambda x: len(x.que), shuffle=True)
    test_iter = BucketIterator(test, batch_size=batch_size, \
            device=device, sort_key=lambda x: len(x.que), shuffle=True)
    
    if hp.embedding!=None:
        print("Loading pretrained embeddings")
        inp_lang.vocab.load_vectors(hp.embedding)
        opt_lang.vocab.load_vectors(hp.embedding)

    return train_iter, val_iter,test_iter, inp_lang, opt_lang
