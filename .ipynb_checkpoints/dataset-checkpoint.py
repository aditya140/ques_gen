# from torchtext test cases
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k



def load_question_dataset(batch_size, device=0):
    spacy_en = spacy.load('en')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    inp_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')
    opt_lang = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

    

    

def load_dataset(batch_size, device=0):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    DE = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))

    print(train.src)
    DE.build_vocab(train.src)
    EN.build_vocab(train.trg)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=device, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN