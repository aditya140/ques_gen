from torch.optim import Adam

import hyperparams as hp
from datasets import load_question_dataset
from models import Encoder, Decoder, Seq2Seq
from sgdr import SGDRScheduler
from utils import train
from evaluate import evaluate_metrics

train_iter, val_iter, test_iter, inp_lang, opt_lang = load_question_dataset(batch_size=hp.batch_size, dataset=hp.dataset, device=hp.device)

use_pretrained=False
if hp.embedding==None:
    use_pretrained=True

encoder = Encoder(source_vocab_size=len(inp_lang.vocab),
                  embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                  n_layers=hp.n_layers, dropout=hp.dropout,use_pretrained=use_pretrained)
decoder = Decoder(target_vocab_size=len(opt_lang.vocab),
                  embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                  n_layers=hp.n_layers, dropout=hp.dropout,use_pretrained=use_pretrained)
seq2seq = Seq2Seq(encoder, decoder)

seq2seq.to(hp.device)
optimizer = Adam(seq2seq.parameters(), lr=hp.max_lr)
scheduler = SGDRScheduler(optimizer, max_lr=hp.max_lr, cycle_length=hp.cycle_length)

train(seq2seq, optimizer, scheduler, train_iter, val_iter, num_epochs=hp.num_epochs)

evaluate_metrics(seq2seq,test_iter)