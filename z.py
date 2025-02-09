import data_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='deepspeech', choices=['whisper_tiny', 'deepspeech'])
parser.add_argument('--dataset', type=str, default='LibriSpeech', choices=['LibriSpeech'])

args = parser.parse_args()

TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)
batch = next(iter(train_loader))
features, targets, input_lengths, target_lengths  = batch
