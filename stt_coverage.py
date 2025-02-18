from models.deepspeech import DeepSpeechModel
import constants
import os
import data_loader
from types import SimpleNamespace
import torch
import tool
from coverage import *

path = os.path.join(constants.PRETRAINED_MODELS, 'LibriSpeech/deepspeech.pth')
model = DeepSpeechModel(model_path=path)

args = SimpleNamespace()
args.dataset = "LibriSpeech"
args.batch_size = 32

TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)

sample_rate = 16000
duration = 5
random_audio = torch.randn(1, sample_rate * duration)

spec = model.transform(random_audio)

layer_size_dict = tool.get_layer_output_sizes(model.model, spec)

num_neuron = 0
for layer_name in layer_size_dict.keys():
    num_neuron += layer_size_dict[layer_name][0]
print('Total %d layers: ' % len(layer_size_dict.keys()))
print('Total %d neurons: ' % num_neuron)

print("-----------NC----------")
# nc = NC(model.model, layer_size_dict, hyper=0.5)
# nc.assess(train_loader)
# train_coverage = nc.current
# print(f"Train coverage: {train_coverage}")
# nc.assess(test_loader)
# test_coverage = nc.current
# print(f"Test coverage {test_coverage}")
# print(f"Increase: {test_coverage-train_coverage}")
print("-------------KMNC-----------")
kmnc = KMNC(model.model, layer_size_dict, hyper=10)
kmnc.assess(train_loader)
train_coverage = kmnc.current
print(f"Train coverage: {train_coverage}")

kmnc.assess(test_loader)
test_coverage = kmnc.current
print(f"Test coverage: {test_coverage}")


# nlc = NLC(model.model, layer_size_dict)
# nlc.assess(train_loader)
# train_coverage = nlc.current
# print(f"Train coverage: {train_coverage}")

# nlc.assess(test_loader)
# test_coverage = nlc.current
# print(f"Test coverage: {test_coverage}")
