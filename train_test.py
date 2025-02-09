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
    "DeepSpeech",
    "asr"
]

# --- Global settings and mapping ---
sample_rate = 16000
n_mels = 128

# Define a simple alphabet and mapping.
# We reserve index 0 for the CTC blank.
alphabet = " abcdefghijklmnopqrstuvwxyz'"
char_map = {c: i + 1 for i, c in enumerate(alphabet)}
int_to_char = {i: c for c, i in char_map.items()}
n_hidden = 2048
n_class = len(alphabet) + 1  # plus one for the CTC blank

def transcript_to_int(transcript):
    """Convert transcript string to list of integer labels."""
    transcript = transcript.lower()
    return [char_map[c] for c in transcript if c in char_map]

# Create a MelSpectrogram transform.
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=n_mels
)

# --- Data Collation ---
def collate_fn(batch):
    """
    Processes a batch of LibriSpeech samples.
    Each sample is a tuple (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id).
    This function:
       1. Converts multi-channel audio to mono.
       2. Resamples audio if necessary.
       3. Computes a MelSpectrogram and transposes it to (time, n_mels).
       4. Pads the features along the time dimension.
       5. Concatenates target label sequences and returns their lengths.
    """
    features = []
    targets = []
    input_lengths = []
    target_lengths = []
    
    for waveform, sr, transcript, *_ in batch:
        # Convert multi-channel to mono if needed.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample (if the sample rate is not the desired one)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        # Compute MelSpectrogram; output shape: (1, n_mels, time)
        mel_spec = mel_transform(waveform)
        # Rearrange to (1, time, n_mels)
        mel_spec = mel_spec.transpose(1, 2)
        # Remove channel dimension (since all audio is mono now) → (time, n_mels)
        mel_spec = mel_spec.squeeze(0)
        features.append(mel_spec)
        input_lengths.append(mel_spec.shape[0])
        
        # Convert transcript into a tensor of ints.
        t = torch.tensor(transcript_to_int(transcript), dtype=torch.long)
        targets.append(t)
        target_lengths.append(len(t))
    
    # Pad the feature sequences to the maximum time length in the batch.
    features = pad_sequence(features, batch_first=True)  # shape: (batch, max_time, n_mels)
    # Add a channel dimension so final shape becomes (batch, 1, max_time, n_mels)
    features = features.unsqueeze(1)
    
    # Concatenate target sequences into a flat 1D tensor (required by CTCLoss).
    targets = torch.cat(targets)
    
    return features, targets, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(target_lengths, dtype=torch.long)

# --- Greedy Decoder ---
def greedy_decoder(output):
    """
    Decodes model outputs (log probabilities) by taking the argmax at each time step,
    then collapsing repeated characters and removing blanks (index 0).
    Input: output tensor of shape (batch, time, n_class).
    Returns: list of decoded strings.
    """
    # For each time step choose highest-probability token.
    pred_tokens = output.argmax(dim=2)
    results = []
    for tokens in pred_tokens:
        prev = None
        res = []
        for t in tokens:
            t = t.item()
            if t != prev and t != 0:  # skip repeated characters and blank(0)
                res.append(t)
            prev = t
        text = ''.join([int_to_char.get(i, '') for i in res])
        results.append(text)
    return results

# --- Training and Testing Functions ---
def train_deepspeech(dataset_name, device):
    # Hyperparameters
    batch_size = 4
    epochs = 3  # For demonstration, we run one epoch. Increase as needed.
    learning_rate = 1e-4

    if dataset_name == "LibriSpeech":
        train_dataset = torchaudio.datasets.LIBRISPEECH("./datasets/", url="train-clean-100", download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = DeepSpeech(n_feature=n_mels, n_hidden=n_hidden, n_class=n_class, dropout=0.0)
    path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (dataset_name, "deepspeech")))
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # The model outputs log probabilities; CTCLoss expects these.
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for batch_idx, (features, targets, input_lengths, target_lengths) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
            
            optimizer.zero_grad()
            # Forward pass: model expects input shape (batch, channel, time, feature)
            outputs = model(features)  # returns (batch, time, n_class)
            # For CTCLoss, transpose to shape (time, batch, n_class)
            outputs = outputs.transpose(0, 1)
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(features), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        avg_loss = total_loss / len(train_loader)
        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_loss))
    
    path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (dataset_name, "deepspeech")))
    torch.save(model.state_dict(), path)

def test_deepspeech(dataset_name, device):
    batch_size = 4
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    model = DeepSpeech(n_feature=n_mels, n_hidden=n_hidden, n_class=n_class, dropout=0.0)
    path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (dataset_name, "deepspeech")))
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

    if dataset_name == "LibriSpeech":
        test_dataset = torchaudio.datasets.LIBRISPEECH("./datasets/", url="test-clean", download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    test_loss = 0
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in test_loader:
            features, targets = features.to(device), targets.to(device)
            input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
            outputs = model(features)  # shape: (batch, time, n_class)
            outputs_for_loss = outputs.transpose(0, 1)  # shape: (time, batch, n_class)
            loss = criterion(outputs_for_loss, targets, input_lengths, target_lengths)
            test_loss += loss.item()
            
            preds = greedy_decoder(outputs)
            all_predictions.extend(preds)
            
            # Reconstruct ground truth texts from the flat targets using target_lengths.
            idx = 0
            for length in target_lengths.cpu().tolist():
                seq = targets[idx:idx+length]
                text = ''.join([int_to_char.get(t.item(), '') for t in seq])
                all_ground_truths.append(text)
                idx += length

    test_loss /= len(test_loader)
    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    print("Sample predictions:")
    for pred, truth in zip(all_predictions[:5], all_ground_truths[:5]):
        print("Predicted: '{}' | Ground Truth: '{}'".format(pred, truth))

def train(model_name, dataset_name, device):
    if model_name == "deepspeech":
        train_deepspeech(dataset_name, device)

def test(model_name, dataset_name, device):
    if model_name == "deepspeech":
        test_deepspeech(dataset_name, device)

def asr(model_name, dataset, pretrained=False, progress=True, device="cpu", **kwargs):
    if model_name == "deepspeech":
        model = DeepSpeech(n_feature=n_mels, n_hidden=n_hidden, n_class=n_class, dropout=0.0)
        if pretrained:
            path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (dataset, model_name)))
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
            model.to(device)
    return model
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test ASR model")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--model-name", type=str, choices=["deepspeech"], required=True, help="Model to use")
    parser.add_argument("--dataset", type=str, choices=["LibriSpeech"], required=True, help="Dataset to use")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print(f"Training {args.model_name} on {args.dataset} dataset...")
        train(args.model_name, args.dataset, device)
    if args.test:
        print(f"Testing {args.model_name} on {args.dataset} dataset...")
        test(args.model_name, args.dataset, device)
    
    if not args.train and not args.test:
        print("Specify at least --train or --test.")
    
