import torch
import torchaudio
import torchaudio.transforms as T


sample_rate = 16000
n_mels = 128
n_fft = 400
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=n_mels,
    n_fft=n_fft
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
waveform, sr = torchaudio.load("audio.wav")
print(waveform.shape)
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
print(waveform.shape)
spec = mel_transform(waveform).transpose(1, 2)
print(spec.shape)


inverse_mel_scale = T.InverseMelScale(
    n_stft=n_fft // 2 + 1,
    n_mels=n_mels,
    sample_rate=sample_rate
)

griffin_lim = T.GriffinLim(
    n_fft=n_fft,
    power=1.0,  # Use magnitude spectrogram
)

lspec = inverse_mel_scale(spec.transpose(1,2))
print(lspec.shape)
w = griffin_lim(lspec)
print(w.shape)
torchaudio.save('modified_audio.wav', w, sample_rate)
