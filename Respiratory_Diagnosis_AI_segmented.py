# Fix for OpenMP conflict between PyTorch and librosa
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sounddevice as sd


# ─────────────────────────────────────────────
# Helper: slice a full audio array into non-overlapping 5-second segments.
# Returns a list of fixed-length numpy arrays.
# ─────────────────────────────────────────────
def slice_into_segments(audio, sample_rate, duration):
    """
    Split `audio` into non-overlapping segments of `duration` seconds.
    The last segment is zero-padded if it is shorter than `duration` seconds.
    Returns a list of 1-D numpy arrays, each of length sample_rate * duration.
    """
    segment_len = sample_rate * duration
    segments = []
    start = 0
    while start < len(audio):
        chunk = audio[start : start + segment_len]
        if len(chunk) < segment_len:                        # pad the final short chunk
            chunk = np.pad(chunk, (0, segment_len - len(chunk)))
        segments.append(chunk)
        start += segment_len
    return segments


# ─────────────────────────────────────────────
# Helper: compute MFCC tensor from a single audio segment.
# ─────────────────────────────────────────────
def segment_to_mfcc(segment, sample_rate, n_mfcc):
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # (1, n_mfcc, time)


# ─────────────────────────────────────────────
# Preview: plays one random file from the dataset.
# ─────────────────────────────────────────────
def preview_random_sample(dataset, categories, sample_rate, duration):
    idx = random.randint(0, len(dataset.file_samples) - 1)
    # dataset.samples stores (filepath, label) tuples at the FILE level
    fpath, label = dataset.file_samples[idx]

    print("=" * 45)
    print("🎧 Random Audio Sample Preview")
    print("=" * 45)
    print(f"  Index     : {idx}")
    print(f"  Label #   : {label}")
    print(f"  Category  : {categories[label]}")
    print(f"  File      : {os.path.basename(fpath)}\n")

    # Load the whole file for display / playback (no duration cap here)
    audio, sr = librosa.load(fpath, sr=sample_rate)

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Sample {idx} — {categories[label]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    print("  ▶ Playing audio...")
    sd.play(audio, samplerate=sr)
    sd.wait()

    input("  Press ENTER to continue to training...")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
#
# KEY CHANGE: each item in __getitem__ is now ONE 5-second SEGMENT, not one
# whole file.  This means a 20-second file contributes 4 training samples and
# a 60-second file contributes 12 — all sharing the same diagnosis label.
# ─────────────────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, audio_dir, diagnosis_csv, sample_rate, duration, n_mfcc, label2idx):
        self.sample_rate = sample_rate
        self.duration    = duration
        self.n_mfcc      = n_mfcc

        # Map patient_id → diagnosis label index
        df = pd.read_csv(diagnosis_csv, header=None, names=['patient_id', 'diagnosis'])
        patient_labels = dict(zip(df['patient_id'].astype(str), df['diagnosis']))

        # file_samples  → (filepath, label)  — used by preview_random_sample
        # samples       → (filepath, start_sample, label)  — one entry per segment
        self.file_samples = []
        self.samples      = []

        for fname in os.listdir(audio_dir):
            if not fname.endswith('.wav'):
                continue
            patient_id = fname.split('_')[0]
            diagnosis  = patient_labels.get(patient_id)
            if diagnosis not in label2idx:
                continue

            fpath = os.path.join(audio_dir, fname)
            label = label2idx[diagnosis]
            self.file_samples.append((fpath, label))

            # Determine how many segments this file produces without loading
            # the entire file into RAM here — we just need its duration.
            file_duration = librosa.get_duration(path=fpath)
            segment_len   = sample_rate * duration
            num_segments  = max(1, int(np.ceil(file_duration / duration)))

            for seg_idx in range(num_segments):
                start_sample = seg_idx * segment_len
                self.samples.append((fpath, start_sample, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, start_sample, label = self.samples[idx]
        segment_len = self.sample_rate * self.duration

        # Load only the 5-second window we need (offset + duration)
        offset_seconds = start_sample / self.sample_rate
        audio, _ = librosa.load(
            fpath,
            sr       = self.sample_rate,
            offset   = offset_seconds,
            duration = self.duration
        )

        # Zero-pad if the tail segment is short
        if len(audio) < segment_len:
            audio = np.pad(audio, (0, segment_len - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label


# ─────────────────────────────────────────────
# Model — unchanged from original
# ─────────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self, num_classes, n_mfcc, time_frames):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        H = n_mfcc      // 4   # two MaxPool2d halvings on height
        W = time_frames // 4   # two MaxPool2d halvings on width
        self.fc1 = nn.Linear(64 * H * W, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────
# Training loop — unchanged
# ─────────────────────────────────────────────
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"  loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


# ─────────────────────────────────────────────
# Test loop — unchanged
# ─────────────────────────────────────────────
def test_loop(dataloader, model, loss_fn):
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred       = model(X)
            test_loss += loss_fn(pred, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct   /= size
    print(f"  Test Error — Accuracy: {100*correct:.1f}%,  Avg loss: {test_loss:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Whole-file inference with soft voting across segments
#
# How it works:
#   1. Load the entire audio file.
#   2. Slice it into non-overlapping 5-second segments.
#   3. Run each segment through the model → raw logits.
#   4. Convert logits to probabilities (softmax).
#   5. Average the probabilities across all segments  ← "soft voting"
#   6. The class with the highest average probability is the final prediction.
#
# Soft voting is preferred over hard (majority) voting because it weighs
# confident segment predictions more heavily than uncertain ones.
# ─────────────────────────────────────────────────────────────────────────────
def predict_file(fpath, model, sample_rate, duration, n_mfcc, categories):
    model.eval()
    audio, _ = librosa.load(fpath, sr=sample_rate)   # load full file
    segments  = slice_into_segments(audio, sample_rate, duration)

    print(f"\n  File      : {os.path.basename(fpath)}")
    print(f"  Length    : {len(audio)/sample_rate:.1f}s  →  {len(segments)} segment(s) of {duration}s")

    seg_probs = []  # will hold one probability vector per segment
    with torch.no_grad():
        for i, seg in enumerate(segments):
            mfcc   = segment_to_mfcc(seg, sample_rate, n_mfcc).unsqueeze(0)  # (1,1,H,W)
            logits = model(mfcc)                                               # (1, num_classes)
            probs  = F.softmax(logits, dim=1).squeeze(0).numpy()              # (num_classes,)
            seg_probs.append(probs)

            seg_pred = categories[np.argmax(probs)]
            print(f"    Segment {i+1:>2d}: {seg_pred:<16s}  "
                  f"(confidence {100*probs.max():.1f}%)")

    # Soft vote: average probability across all segments
    avg_probs      = np.mean(seg_probs, axis=0)         # (num_classes,)
    final_class_idx = int(np.argmax(avg_probs))
    final_label     = categories[final_class_idx]

    print(f"\n  ── Soft-vote result ──────────────────────")
    for i, (cat, p) in enumerate(zip(categories, avg_probs)):
        bar = "█" * int(p * 30)
        marker = " ◄" if i == final_class_idx else ""
        print(f"    {cat:<16s} {100*p:5.1f}%  {bar}{marker}")
    print(f"\n  FINAL PREDICTION: {final_label}  "
          f"(avg confidence {100*avg_probs[final_class_idx]:.1f}%)")
    print("  ──────────────────────────────────────────\n")

    return final_label, avg_probs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # ── Paths ── update these to match your machine ──────────────────────────
    AUDIO_DIR     = r"C:\Users\owenw\OneDrive - purdue.edu\Desktop\respiratory-detector\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
    DIAGNOSIS_CSV = r"C:\Users\owenw\OneDrive - purdue.edu\Desktop\respiratory-detector\Respiratory_Sound_Database\Respiratory_Sound_Database\patient_diagnosis.csv"

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    SAMPLE_RATE   = 22050
    DURATION      = 2        # seconds per segment
    N_MFCC        = 40
    BATCH_SIZE    = 32
    EPOCHS        = 10
    LEARNING_RATE = 1e-3
    CATEGORIES    = ['COPD', 'Healthy', 'URTI', 'Bronchiectasis',
                     'Pneumonia', 'Bronchiolitis', 'Asthma', 'LRTI']
    label2idx     = {label: idx for idx, label in enumerate(CATEGORIES)}

    # ── Build dataset (segments) ──────────────────────────────────────────────
    print("Building dataset from segments...")
    dataset = AudioDataset(AUDIO_DIR, DIAGNOSIS_CSV, SAMPLE_RATE, DURATION, N_MFCC, label2idx)
    print(f"  Total segments in dataset: {len(dataset)}")
    print(f"  (from {len(dataset.file_samples)} audio files)\n")

    # ── Optional audio preview ────────────────────────────────────────────────
    preview_random_sample(dataset, CATEGORIES, SAMPLE_RATE, DURATION)

    # ── Train / test split ────────────────────────────────────────────────────
    train_size   = int(0.8 * len(dataset))
    test_size    = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_data,     batch_size=BATCH_SIZE)

    # ── Build model ───────────────────────────────────────────────────────────
    time_frames = (SAMPLE_RATE * DURATION) // 512 + 1
    model    = NeuralNet(num_classes=len(CATEGORIES), n_mfcc=N_MFCC, time_frames=time_frames)
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training ──────────────────────────────────────────────────────────────
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n{'─'*35}")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader,   model, loss_fn)
    print("Training complete!\n")

    # ── Whole-file soft-vote inference demo ───────────────────────────────────
    # Pick a random file from the dataset and classify it end-to-end.
    demo_fpath, true_label = random.choice(dataset.file_samples)
    print("=" * 45)
    print("🔬 Whole-file Soft-Vote Inference Demo")
    print("=" * 45)
    print(f"  True label: {CATEGORIES[true_label]}")
    predict_file(demo_fpath, model, SAMPLE_RATE, DURATION, N_MFCC, CATEGORIES)


if __name__ == "__main__":
    main()
