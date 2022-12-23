import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        # Define Optimizer, Loss Function, and Evaluation Metric
        self.criterion = nn.MSELoss().cuda() # always to cuda
        # self.epoch_loss = 0

        # use RMSE as evaluation metric. USe MSE then pass argument False
        self.metric  = torchmetrics.MeanSquaredError(squared=False)
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, x):
        # set initial states
        h_n, c_n = self.encoder(x)
        # print(f"Encoder output: {h_n.shape}, {c_n.shape}")
        # create initial input (start with zeros)
        decoder_input = torch.zeros(x.size(0), 1, 1).cuda() # always to cuda
        # print(decoder_input.device)
        # init output tensor
        # print(x.size(1), x.size(0))
        outputs = torch.zeros(x.size(1), x.size(0),  1)
        # print(outputs.device)
        # decode hidden state of last time step
        for t in range(x.size(1)):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, h_n, c_n)
            # print(f"Decoder output: {decoder_output.shape}, {h_n.shape}, {c_n.shape}")
            # decoder_output = decoder_output.squeeze(1)
            outputs[t] = decoder_output.squeeze(1)
            decoder_input = decoder_output
        return outputs


    def training_step(self, batch, batch_idx):
        # get input and targets and get to cuda. Remember, input is on [:,:,0] and [:,:,2], while target is on [:,:,1]
        inp = batch[:, :, [0,2]]
        target = batch[:, :, 1]
        # forward prop
        output = self.forward(inp)
        # output = [batch size, seq len, output dim]
        # target = [batch size, seq len, output dim]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim).cuda() # always to cuda
        target = target.contiguous().view(-1, output_dim).cuda()
        # print(output.device, target.device)
        # loss and backprop
        loss = self.criterion(output, target)
        # record loss
        # self.epoch_loss += loss.item()

        rmse_value = self.metric(output, target)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {"loss": loss, "rmse_value": rmse_value}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse_value"] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_train_rmse', avg_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    
