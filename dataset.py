# from torchtext test cases
import spacy
from torchtext.data import Field, BucketIterator,TabularDataset
import glob
import json
import pandas as pd
import unicodedata
import re
from sklearn.model_selection import train_test_split


class QGenDataset(object):
    def __init__(self):
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

    def _fetch_data(self,normalize=True):
        self.ans_que=self._get_dataset(normalize=normalize)
        data=pd.DataFrame(self.ans_que,columns=["ans","que"])
        data=data.drop_duplicates()
        self.ans_que=data[["ans","que"]].values
        for i in self.ans_que:
            self.ans.append(i[0])
            self.que.append(i[1])

    # def _split(self,train=70, test=15, val=15):
    #     self.ans_train=self.ans[0:int((len(self.ans)*train)/100)]
    #     self.ans_train=self.ans[int((len(self.ans)*train)/100):int((len(self.ans)*test)/100)]
    #     self.ans_train=self.ans[int((len(self.ans)*test)/100):int((len(self.ans)*(100-train-test))/100)]
        
    #     self.que_train=self.que[0:int((len(self.que)*train)/100)]
    #     self.que_train=self.que[int((len(self.que)*train)/100):int((len(self.que)*test)/100)]
    #     self.que_train=self.que[int((len(self.que)*test)/100):int((len(self.que)*(100-train-test))/100)]

    #     for i in range(len(self.ans_train)):
    #         self.ans_que_train.append([self.ans_train[0],self.ans_train[1]])
        
    #     for i in range(len(self.ans_test)):
    #         self.ans_que_test.append([self.ans_test[0],self.ans_test[1]])
        
    #     for i in range(len(self.ans_val)):
    #         self.ans_que_val.append([self.ans_val[0],self.ans_val[1]])

    #     return self.ans_que_train,self.ans_que_test,self.ans_que_val
        
    def to_csv(self):
        raw_data = {'ans' : [line for line in self.ans], 'que': [line for line in self.que]}
        df = pd.DataFrame(raw_data, columns=["ans", "que"])
        # remove very long sentences and sentences where translations are 
        # not of roughly equal length
        df['ans_len'] = df['ans'].str.count(' ')
        df['que_len'] = df['que'].str.count(' ')
        df = df.query('ans_len < 80 & que_len < 80')
        df = df.drop_duplicates()
        train, val = train_test_split(df, test_size=0.15)
        train.to_csv("./.data/train.csv", index=False)
        val.to_csv("./.data/val.csv", index=False)

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
                        if not normalize:
                            res.append(self._normalize(self._get_sentence(para["context"],qa["answers"][0]["answer_start"],qa["answers"][0]["text"])))
                            res.append(self._normalize(qa["question"]))
                        else:
                            res.append(self._get_sentence(para["context"],qa["answers"][0]["answer_start"],qa["answers"][0]["text"]))
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





def load_question_dataset(batch_size, device=0):
    spacy_en = spacy.load('en')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    inp_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')
    opt_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

    dataset=QGenDataset()
    dataset._fetch_data(normalize=True)
    dataset.to_csv()

    # associate the text in the 'English' column with the EN_TEXT field, # and 'French' with FR_TEXT
    data_fields = [('ans', inp_lang), ('que', opt_lang)]
    train,val = TabularDataset.splits(path='./.data/', train='train.csv', validation='val.csv', format='csv', fields=data_fields)



    inp_lang.build_vocab(train,val)
    opt_lang.build_vocab(train,val)

    train_iter = BucketIterator(train, batch_size=batch_size, \
            device=device, repeat=False , sort_key=lambda x: len(x.que), shuffle=True)
    # train_iter, val_iter = BucketIterator((train, val), batch_size=batch_size, device=device, repeat=False , sort_key=lambda x: len(x.que))
    return train_iter, inp_lang, opt_lang
