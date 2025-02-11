import torch
import torchaudio
import numpy as np
from models.deepspeech import DeepSpeechModel
import torch.optim as optim
import os
import constants

# Load pretrained DeepSpeech model
path = os.path.join(constants.PRETRAINED_MODELS, 'LibriSpeech/deepspeech.pth')
model = DeepSpeechModel(model_path=path)
model.train()

# Generate random audio (5 seconds, 16kHz)
sample_rate = 16000
duration = 5
random_audio = torch.randn(1, sample_rate * duration)

# Target transcription
target_text = "hello world"

# Convert target text to tensor of label indices
# labels = [model.decoder.labels.index(c) for c in target_text if c in model.decoder.labels]
# target = torch.tensor(labels)
target = model.encode(target_text)

# random_audio, sr = torchaudio.load('audio.wav')
# if sr != sample_rate:
#     random_audio = torchaudio.functional.resample(random_audio, sr, sample_rate)
# Prepare audio for input
# random_audio.requires_grad_()
spec = model.transform(random_audio)
print(spec.shape)
# inverted_audio = model.inverse_transform(spec)

# Save the resulting audio
# torchaudio.save('optimized_audio.wav', inverted_audio, sample_rate)

# exit(0)

print(spec.shape)
# Set up optimization
random_audio.requires_grad = True
optimizer = optim.Adam([random_audio], lr=0.01)

# Gradient descent loop
num_iterations = 1000
for i in range(num_iterations):
    optimizer.zero_grad()
    
    # Forward pass
    spec = model.transform(random_audio)
    output = model.model(spec)
    output = torch.nn.functional.log_softmax(output, dim=2)
    
    # Compute CTC loss
    input_lengths = torch.full(size=(1,), fill_value=output.size(1), dtype=torch.long)
    target_lengths = torch.tensor([len(target)], dtype=torch.long)
    loss = torch.nn.CTCLoss()(output.transpose(0, 1), target, input_lengths, target_lengths)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}")

# print(spec.shape)
# Generate final transcription
with torch.no_grad():
    spec = model.transform(random_audio)
    output = model.model(spec)
    decoded = model.decode(output)
    print(f"Final transcription: {decoded}")

# Convert optimized spectrogram back to audio
# inverted_audio = model.inverse_transform(spec.to(torch.device('cpu')))

# Save the resulting audio
torchaudio.save('optimized_audio.wav', random_audio.detach(), sample_rate)
