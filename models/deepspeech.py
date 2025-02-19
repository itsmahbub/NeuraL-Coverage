import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.models import DeepSpeech
import torch.nn.functional as F

class DeepSpeechModel:
    def __init__(self, model_path=None, device=None):
        """
        Wrapper for DeepSpeech model to add utility functions like loading pretrained models,
        decoding, transforming audio to spectrograms, and inverting spectrograms back to audio.
        """
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.labels = " abcdefghijklmnopqrstuvwxyz'"
        self.n_mels = 128
        self.n_fft = 400
        self.n_hidden = 2048
        self.hop_length = 160
        self.n_class = len(self.labels) + 1
        self.sample_rate=16000
        self.char_map = {c: i + 1 for i, c in enumerate(self.labels)}
        self.int_to_char = {i: c for c, i in self.char_map.items()}
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
        )
        
        self.inverse_mel_scale = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate
        )
        
        self.griffin_lim = T.GriffinLim(
            n_fft=self.n_fft,
            power=1.0,  # Use magnitude spectrogram
        )

        self.model = DeepSpeech(n_feature=self.n_mels, n_hidden=self.n_hidden, n_class=self.n_class, dropout=0.0)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()

    def transform(self, audio):
        """
        Converts raw audio waveform to a Mel spectrogram.
        """
        return self.mel_transform(audio).transpose(1, 2).to(self.device)

    def inverse_transform(self, mel_spectrogram):
        """
        Converts a mel spectrogram back to waveform using torchaudio's InverseMelScale and Griffin-Lim.
        Returns:
        - waveform (Tensor): Reconstructed waveform.
        """
        # Step 1: Convert mel spectrogram back to linear spectrogram
        
        linear_spectrogram = self.inverse_mel_scale(mel_spectrogram.transpose(1, 2))

        # Step 2: Apply Griffin-Lim to reconstruct waveform
        
        waveform = self.griffin_lim(linear_spectrogram)

        return waveform

      
    def encode(self, text):
        """
        Encode text to tensor
        """
        text = text.lower()
        return torch.tensor([self.char_map[c] for c in text if c in self.char_map], dtype=torch.long).to(self.device)

    def decode(self, output):
        """
        Decodes the output probabilities from the model into text using greedy decoding.
        """
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
            text = ''.join([self.int_to_char.get(i, '') for i in res])
            results.append(text)
        return results
