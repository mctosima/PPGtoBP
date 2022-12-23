import torch
import torch.nn as nn

"""
Remember, for each subject, we have 3 column signal data.
First is PPG, Second is ABP, Third is ECG.
We want to make a seq2seq model that can predict ABP from PPG and ECG.
"""

# create ABP sequence encoder
class ABPEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(ABPEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hid_dim).cuda() # always to cuda
        # print(h0.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hid_dim).cuda() # always to cuda
        # print(c0.device)
        # forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        # return the final hidden state and cell state
        return h_n, c_n

