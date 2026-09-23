import sys
import os
import torch
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Loading '{MODEL_ID}' on {device}...")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device)
model.eval()

def test_file(file_path):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        data, sr = sf.read(file_path, dtype='float32')
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
    except Exception:
        waveform, sr = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 16000:
        waveform = T.Resample(orig_freq=sr, new_freq=16000)(waveform)

    norm_audio = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
    audio_numpy = norm_audio.squeeze().cpu().numpy()

    inputs = extractor(audio_numpy, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    fake_prob = probs[1] * 100
    real_prob = probs[0] * 100

    print("=" * 55)
    print(f"Target: {file_path}")
    print(f"AI/Synthetic Probability:  {fake_prob:.2f}%")
    print(f"Human Probability:         {real_prob:.2f}%")
    print(f"Verdict: {'CRITICAL AI CLONE' if fake_prob >= 50.0 else 'AUTHENTIC HUMAN SPEECH'}")
    print("=" * 55)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "1006.wav"
    test_file(target)