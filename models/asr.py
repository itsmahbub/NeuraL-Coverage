import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models import DeepSpeech
import constants
import os
import argparse

__all__ = [
    "deepspeech",
]

n_mels = 128
alphabet = " abcdefghijklmnopqrstuvwxyz'"
n_hidden = 2048
n_class = len(alphabet) + 1  # plus one for the CTC blank


def deepspeech(pretrained=False, dataset=None, progress=True, device="cpu", **kwargs):
    print("Device: ", device)
    model = DeepSpeech(n_feature=n_mels, n_hidden=n_hidden, n_class=n_class, dropout=0.0)
    if pretrained:
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (dataset, "deepspeech")))
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.to(device)
    return model
        

if __name__ == "__main__":
   pass
