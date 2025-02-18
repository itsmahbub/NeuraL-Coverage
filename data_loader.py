import os
import random
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# import torchvision.transforms as transforms
# import torchtext
# from torchtext import data
# from torchtext import datasets
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.vocab import vocab as build_vocab, GloVe
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import VocabTransform, Sequential, ToTensor

import constants


class CIFAR10Dataset(Dataset):
    def __init__(self,
                 args,
                 image_dir=constants.CIFAR10_JPEG_DIR,
                 split='train'):
        super(CIFAR10Dataset).__init__()
        assert split in ['train', 'test']
        self.total_class_num = 10
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
                ])

        self.image_list = []
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        label = self.class_list.index(label)
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label

class ImageNetDataset(Dataset):
    def __init__(self,
                 args,
                 image_dir=constants.IMAGENET_JPEG_DIR,
                 label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                 split='train'):
        super(ImageNetDataset).__init__()
        assert split in ['train', 'val']
        self.total_class_num = 1000
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

class DataLoader(object):
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.init_param()

    def init_param(self):
        self.gpus = 1
        # self.gpus = torch.cuda.device_count()
        # TODO: multi GPU

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle,
                            **self.kwargs
                        )
        return data_loader

def get_loader(args):
    assert args.dataset in ['CIFAR10', 'ImageNet', 'IMDB', 'LibriSpeech']
    if args.dataset == 'CIFAR10':
        train_data = CIFAR10Dataset(args, split='train')
        test_data = CIFAR10Dataset(args, split='test')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 10
    elif args.dataset == 'ImageNet':
        train_data = CIFAR10Dataset(args, split='train')
        test_data = CIFAR10Dataset(args, split='val')
        loader = DataLoader(args)
        train_loader = loader.get_loader(train_data, False)
        test_loader = loader.get_loader(test_data, False)
        seed_loader = loader.get_loader(test_data, True)
        TOTAL_CLASS_NUM = 1000
    elif args.dataset == 'IMDB':
        MAX_VOCAB_SIZE = 25_000

        tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

        def yield_tokens(data_iter):
            for label, line in data_iter:
                yield tokenizer(line)

        # Load IMDB dataset
        train_iter = IMDB(split=('train'))

        # Build vocabulary
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), max_tokens=MAX_VOCAB_SIZE, specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])

        # Load pretrained GloVe embeddings
        vectors = GloVe(name='6B', dim=100)
        vocab.vectors = vectors.get_vecs_by_tokens(vocab.get_itos(), lower_case_backup=True)

        def collate_batch(batch):
            text_list, label_list = [], []
            for label, text in batch:
                tokenized_text = tokenizer(text)
                indexed_text = vocab(tokenized_text)[:constants.PAD_LENGTH]  # Truncate if needed
                text_list.append(torch.tensor(indexed_text))
                label_list.append(torch.tensor(label))
            
            text_list = torch.nn.utils.rnn.pad_sequence(text_list, padding_value=vocab["<pad>"])
            label_list = torch.tensor(label_list)
            return text_list, label_list

        # Create DataLoaders
        train_iter, test_iter = IMDB(split=('train', 'test'))  # Reload since iterator was exhausted
        train_loader = torch.utils.data.DataLoader(list(train_iter), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        test_loader = torch.utils.data.DataLoader(list(test_iter), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        seed_loader = torch.utils.data.DataLoader(list(test_iter), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        TOTAL_CLASS_NUM = 2
    elif args.dataset == "LibriSpeech":
        from models.deepspeech import DeepSpeechModel
        model = DeepSpeechModel()
        # sample_rate = 16000
        # n_mels = 128
        # mel_transform = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_mels=n_mels
        # )
        # alphabet = " abcdefghijklmnopqrstuvwxyz'"
        # char_map = {c: i + 1 for i, c in enumerate(alphabet)}

        # def transcript_to_int(transcript):
        #     transcript = transcript.lower()
        #     return [char_map[c] for c in transcript if c in char_map]

        def collate_fn(batch):
            features = []
            targets = []
            input_lengths = []
            target_lengths = []

            for waveform, sr, transcript, *_ in batch:
                # Convert multi-channel to mono if needed.
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # Resample (if the sample rate is not the desired one)
                if sr != model.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                # Compute MelSpectrogram; output shape: (1, n_mels, time)
                mel_spec = model.mel_transform(waveform)
                # Rearrange to (1, time, n_mels)
                mel_spec = mel_spec.transpose(1, 2)
                # Remove channel dimension (since all audio is mono now) → (time, n_mels)
                mel_spec = mel_spec.squeeze(0)
                features.append(mel_spec)
                input_lengths.append(mel_spec.shape[0])
                
                # Convert transcript into a tensor of ints.
                t = torch.tensor(model.encode(transcript), dtype=torch.long)
                targets.append(t)
                target_lengths.append(len(t))
            
            # Pad the feature sequences to the maximum time length in the batch.
            features = pad_sequence(features, batch_first=True)  # shape: (batch, max_time, n_mels)
            # Add a channel dimension so final shape becomes (batch, 1, max_time, n_mels)
            features = features.unsqueeze(1)
            
            # Concatenate target sequences into a flat 1D tensor (required by CTCLoss).
            targets = torch.cat(targets)
            
            return features, targets, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(target_lengths, dtype=torch.long)

        
        train_dataset = torchaudio.datasets.LIBRISPEECH("./datasets/", url="train-clean-100", download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataset = torchaudio.datasets.LIBRISPEECH("./datasets/", url="test-clean", download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        seed_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        TOTAL_CLASS_NUM = len(model.labels) + 1
    return TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader

class FuzzDataset:
    def __init__(self):
        raise NotImplementedError

    def label2index(self):
        raise NotImplementedError

    def get_len(self):
        return len(self.image_list)

    def get_item(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # label = self.cat_list.index(label)
        index = self.label2index(label)
        assert int(index) < self.args.num_class
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return (image, index)

    def build(self):
        image_list = []
        label_list = []
        for i in tqdm(range(self.get_len())):
            (image, label) = self.get_item(i)
            image_list.append(image)
            label_list.append(label)
        return image_list, label_list

    def to_numpy(self, image_list, is_image=True):
        image_numpy_list = []
        for i in tqdm(range(len(image_list))):
            image = image_list[i]
            if is_image:
                image_numpy = image.transpose(0, 2).numpy()
            else:
                image_numpy = image.numpy()
            image_numpy_list.append(image_numpy)
        print('Numpy: %d' % len(image_numpy_list))
        return image_numpy_list

    def to_batch(self, data_list, is_image=True):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.args.batch_size == 0:
                batch_list.append(torch.stack(batch, 0))
                batch = []
            batch.append(self.norm(data) if is_image else data)
        if len(batch):
            batch_list.append(torch.stack(batch, 0))
        print('Batch: %d' % len(batch_list))
        return batch_list

class CIFAR10FuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.CIFAR10_JPEG_DIR,
                 split='test'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                ])
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.image_list = []
        
        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.class_list.index(label_name)

class ImageNetFuzzDataset(FuzzDataset):
    def __init__(self,
                 args,
                 image_dir=constants.IMAGENET_JPEG_DIR,
                 label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                 split='val'):
        self.args = args
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(self.args.image_size),
                        transforms.CenterCrop(self.args.image_size),
                        transforms.ToTensor(),
                ])
        self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index_dict = json.load(f)

        self.class_list = sorted(os.listdir(self.image_dir))[:self.args.num_class]
        for class_name in self.class_list:
            name_list = sorted(os.listdir(self.image_dir + class_name))[:self.args.num_per_class]
            self.image_list += [self.image_dir + class_name + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def label2index(self, label_name):
        return self.label2index_dict[label_name]


if __name__ == '__main__':
    pass


