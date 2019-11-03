import torch

# vocab
pad_idx = 1
sos_idx = 2

# architecture
hidden_dim = 600
embed_dim = 300
n_layers = 2
dropout = 0.3
batch_size = 32
num_epochs = 10

# training
max_lr = 1.2e-4
cycle_length = 3000

# generation
max_len = 100

# system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#dataset
dataset=["squad"]#datatset=["squad","marco"]
