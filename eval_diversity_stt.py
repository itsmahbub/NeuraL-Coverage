import os
import copy
import random
import argparse
from tqdm import tqdm
import torchaudio
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F


import data_loader
import utility
import models
import tool
import coverage
import constants

def get_random_audio():
    sampling_rate = 16000       # Sampling rate: 16 kHz
    duration = 10             # Duration in seconds
    num_samples = int(sampling_rate * duration)

    # Generate random audio data (a tensor of size [num_samples])
    random_audio = torch.randn(1, num_samples)
    return random_audio

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=128
)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='deepspeech', choices=['whisper_tiny', 'deepspeech'])
parser.add_argument('--dataset', type=str, default='LibriSpeech', choices=['LibriSpeech'])
parser.add_argument('--criterion', type=str, default='NC', 
                    choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC'])
parser.add_argument('--output_dir', type=str, default='./test_folder')
# parser.add_argument('--nc', type=int, default=3)
# parser.add_argument('--image_size', type=int, default=32)
# parser.add_argument('--batch_size', type=int, default=100)
# parser.add_argument('--num_workers', type=int, default=4)
# parser.add_argument('--num_class', type=float, default=10)
# parser.add_argument('--num_per_class', type=float, default=5000)
parser.add_argument('--hyper', type=float, default=None)
args = parser.parse_args()
args.exp_name = ('%s-%s-%s' % (args.model, args.criterion, args.hyper))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

utility.make_path(args.output_dir)
fp = open('%s/%s.txt' % (args.output_dir, args.exp_name), 'w')

utility.log(('Model: %s \nCriterion: %s \nHyper-parameter: %s' % (
    args.model, args.criterion, args.hyper
)), fp)

USE_SC = args.criterion in ['LSC', 'DSC', 'MDSC']

model = getattr(models, args.model)(pretrained=True, dataset=args.dataset, device=DEVICE)
# path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))

TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)
# model.to(DEVICE)
model.eval()

random_audio = get_random_audio()
print("Shape: ", random_audio.shape)
# random_audio = torchaudio.functional.resample(random_audio, 16000, 16000)
# random_audio = next(iter(train_loader))[0]
mel_spec = mel_transform(random_audio)
mel_spec = mel_spec.transpose(1, 2)
# mel_spec = mel_spec.squeeze(0)
print("Shape: ", mel_spec.shape)
layer_size_dict = tool.get_layer_output_sizes(model, mel_spec.to(DEVICE))

num_neuron = 0
for layer_name in layer_size_dict.keys():
    num_neuron += layer_size_dict[layer_name][0]
print('Total %d layers: ' % len(layer_size_dict.keys()))
print('Total %d neurons: ' % num_neuron)

if USE_SC:
    criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper, min_var=1e-5, num_class=TOTAL_CLASS_NUM)
else:
    criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper)

criterion.build(train_loader)
if args.criterion not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC']:
    criterion.assess(train_loader)
'''
For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
exceed the memory limit). You can skip this step to speed up the experiment, which
will not affect the conclusion because we only compare the relative order of coverage
values, rather than the exact numbers.
'''
utility.log('Initial coverage: %d' % criterion.current, fp)

criterion1 = copy.deepcopy(criterion)
criterion1.assess(test_loader)
utility.log(('Test: %f, increase: %f' % (criterion1.current, criterion1.current - criterion.current)), fp)
del criterion1

times = 1
for times in [1, 10]:
    criterion2 = copy.deepcopy(criterion)
    for i, (old_text, label, _, _) in enumerate(seed_loader):
        for j in tqdm(range(times * len(list(seed_loader)))):
            text = old_text[torch.randperm(old_text.size()[0])]
            if USE_SC:
                criterion2.step(text.to(DEVICE), text.to(DEVICE))
            else:
                criterion2.step(text.to(DEVICE))
        break
    utility.log(('%s x%d: %f, increase: %f' % (args.dataset, times, criterion2.current, criterion2.current - criterion.current)), fp)
    del criterion2
