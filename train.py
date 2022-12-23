import glob
import torch
import torch.nn as nn
from dataset import *
from lightningmodule import *
from encoder import *
from decoder import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

PATHROOT="dataset/Samples"
FPS=125

# create collate_fn using RNN pad sequence
def collate_fn(data):
    # data is a list of tuples
    # each tuple is (data, label)
    # sort the data list by label
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate data and label
    # data, = zip(*data)
    # merge data (from tuple of 1D tensor to 2D tensor)
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return data

# Initialize weights

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)




def main():
    # populate all files using glob
    files = glob.glob(PATHROOT + "/*.csv")
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    physio_dataset = Dataset(PATHROOT)

    traindl = torch.utils.data.DataLoader(physio_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=8)

    # Model Parameters
    INPUT_DIM = 2
    OUTPUT_DIM = 1
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # create encoder and decoder
    encoder = ABPEncoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = ABPDecoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    encoder.apply(init_weights)
    decoder.apply(init_weights)

    # create seq2seq model
    model = Seq2Seq(encoder, decoder)

    # create callbacks for checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_train_loss",
        dirpath="checkpoints",
        filename="seq2seq-{epoch:02d}-{avg_train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # setup wandb logger
    wandb_logger = WandbLogger(project="seq2seq", log_model=False)

    # pytorch lightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1,
        max_epochs=10,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    # train model
    trainer.fit(model, traindl)


if __name__ == "__main__":
    main()