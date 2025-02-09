from transformers import WhisperForConditionalGeneration

__all__ = [
    "whisper_tiny"
]

def whisper_tiny(pretrained=False, progress=True, device="cpu", **kwargs):
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return model

def draft():
    import torch
    from transformers import WhisperProcessor
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = whisper_tiny()
    model.to(DEVICE)
    model.eval()
    

def data():
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
import torch

def generate_random_audio(sample_rate=16000, duration=1.0):
    # Calculate the number of samples
    num_samples = int(sample_rate * duration)
    
    # Generate random audio data
    random_audio = torch.rand(1, num_samples) * 2 - 1  # Scale to [-1, 1]
    
    return random_audio

# Generate 1 second of random audio at 16kHz
random_audio = generate_random_audio()