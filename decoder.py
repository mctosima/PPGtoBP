import torch
import torch.nn as nn

# create ABP sequence decoder
class ABPDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super(ABPDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
        
    def forward(self, x, h_n, c_n):
        # forward propagate LSTM
        # print(x.shape, h_n.shape, c_n.shape)
        out, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        # decode the hidden state of the last time step
        out = self.fc(out)
        return out, (h_n, c_n)