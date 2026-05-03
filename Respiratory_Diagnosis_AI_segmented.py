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
import soundfile as sf

# ─────────────────────────────────────────────────────────────────────────────
# Helper: slice a full audio array into non-overlapping segments.
# The final chunk is zero-padded if shorter than `duration` seconds.
# Returns a list of 1-D numpy arrays, each of length sample_rate * duration.
# ─────────────────────────────────────────────────────────────────────────────
def slice_into_segments(audio, sample_rate, duration):
    segment_len = sample_rate * duration
    segments = []
    start = 0
    while start < len(audio):
        chunk = audio[start : start + segment_len]
        if len(chunk) < segment_len:
            chunk = np.pad(chunk, (0, segment_len - len(chunk)))
        segments.append(chunk)
        start += segment_len
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute MFCC tensor from a single audio segment.
# Output shape: (1, n_mfcc, time_frames)
# ─────────────────────────────────────────────────────────────────────────────
def segment_to_mfcc(segment, sample_rate, n_mfcc):
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computation: loads each audio file exactly ONCE, slices all its segments
# in RAM, and computes every MFCC up front so training epochs only read from
# a Python list — no disk I/O after this step.
#
# `file_samples` — list of (filepath, label_index) tuples, one per FILE.
#
# Returns:
#   cache — list of (mfcc_tensor, label_index), one entry per segment
#   total — total number of segments cached
# ─────────────────────────────────────────────────────────────────────────────
def precompute_mfccs(file_samples, sample_rate, duration, n_mfcc):
    segment_len = sample_rate * duration

    # Count total segments upfront using fast header-only reads (no decoding)
    total_segments = 0
    for fpath, _ in file_samples:
        info = sf.info(fpath)
        total_segments += max(1, int(np.ceil(info.duration / duration)))

    print(f"    Total segments to pre-compute : {total_segments}")
    print(f"    Across {len(file_samples)} audio files\n")

    cache     = []
    seg_count = 0

    for fpath, label in file_samples:
        # Load the full file once — no repeated seeks per segment
        audio, _ = librosa.load(fpath, sr=sample_rate)

        if len(audio) == 0:
            # Edge case: empty file — insert one silent segment so the label
            # is still represented rather than silently dropped
            silence = np.zeros(segment_len, dtype=np.float32)
            mfcc = librosa.feature.mfcc(y=silence, sr=sample_rate, n_mfcc=n_mfcc)
            cache.append((torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label))
            seg_count += 1
            continue

        # Slice all segments from this file in RAM
        start = 0
        while start < len(audio):
            chunk = audio[start : start + segment_len]
            if len(chunk) < segment_len:
                chunk = np.pad(chunk, (0, segment_len - len(chunk)))

            mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=n_mfcc)
            cache.append((torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label))

            start     += segment_len
            seg_count += 1

            if seg_count % 100 == 0:
                pct = 100 * seg_count / total_segments
                print(f"    [{seg_count:>5d} / {total_segments}]  {pct:.1f}% complete")

    print(f"    Done — {seg_count} segments cached in RAM.\n")
    return cache, seg_count


# ─────────────────────────────────────────────────────────────────────────────
# Preview: loads and plays one random file from the combined dataset.
# ─────────────────────────────────────────────────────────────────────────────
def preview_random_sample(dataset, categories, sample_rate):
    idx = random.randint(0, len(dataset.file_samples) - 1)
    fpath, label = dataset.file_samples[idx]

    print("=" * 45)
    print("🎧  Random Audio Sample Preview")
    print("=" * 45)
    print(f"  Index    : {idx}")
    print(f"  Label    : {categories[label]}  (#{label})")
    print(f"  File     : {os.path.basename(fpath)}\n")

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

    input("  Press ENTER to continue to training...\n")


# ─────────────────────────────────────────────────────────────────────────────
# AudioDataset
#
# Reads the original Respiratory Sound Database where diagnoses come from a
# separate CSV file keyed on patient ID (first token of the filename).
# All segment slicing and MFCC computation happens once in __init__ via
# precompute_mfccs(); __getitem__ simply returns from the in-RAM cache.
# ─────────────────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, audio_dir, diagnosis_csv, sample_rate, duration, n_mfcc, label2idx):
        # Map patient_id → diagnosis string
        df = pd.read_csv(diagnosis_csv, header=None, names=['patient_id', 'diagnosis'])
        patient_labels = dict(zip(df['patient_id'].astype(str), df['diagnosis']))

        # file_samples — one (filepath, label_index) tuple per audio file.
        # Only files whose diagnosis appears in label2idx are included;
        # everything else (e.g. LRTI, Bronchiolitis) is silently skipped.
        self.file_samples = []
        for fname in sorted(os.listdir(audio_dir)):
            if not fname.endswith('.wav'):
                continue
            patient_id = fname.split('_')[0]
            diagnosis  = patient_labels.get(patient_id)
            if diagnosis not in label2idx:
                continue
            self.file_samples.append(
                (os.path.join(audio_dir, fname), label2idx[diagnosis])
            )

        print("  Pre-computing MFCCs for AudioDataset...")
        self.cache, _ = precompute_mfccs(self.file_samples, sample_rate, duration, n_mfcc)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


# ─────────────────────────────────────────────────────────────────────────────
# FolderAudioDataset
#
# Reads a dataset where the label is encoded in the subfolder name, e.g.:
#   root_dir/
#       asthma/   ← matched case-insensitively to CATEGORIES
#           P1AsthmaIU_2.wav
#       healthy/
#           ...
# Any subfolder that does not match a category is skipped with a warning.
# ─────────────────────────────────────────────────────────────────────────────
class FolderAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate, duration, n_mfcc, label2idx):
        # Build a lowercase lookup so folder matching is case-insensitive
        lower_label2idx = {k.lower(): v for k, v in label2idx.items()}

        self.file_samples = []
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label_idx = lower_label2idx.get(folder_name.lower())
            if label_idx is None:
                print(f"  [Warning] Folder '{folder_name}' has no match in CATEGORIES — skipping.")
                continue

            for fname in sorted(os.listdir(folder_path)):
                if not fname.endswith('.wav'):
                    continue
                self.file_samples.append(
                    (os.path.join(folder_path, fname), label_idx)
                )

        print("  Pre-computing MFCCs for FolderAudioDataset...")
        self.cache, _ = precompute_mfccs(self.file_samples, sample_rate, duration, n_mfcc)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


# ─────────────────────────────────────────────────────────────────────────────
# NeuralNet — two conv layers followed by two fully-connected layers.
#
# Input shape : (batch, 1, n_mfcc, time_frames)
# Output shape: (batch, num_classes)  — raw logits, not probabilities
# ─────────────────────────────────────────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self, num_classes, n_mfcc, time_frames):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        # Each MaxPool2d halves both spatial dimensions;
        # two pool layers → divide height and width by 4
        H = n_mfcc      // 4
        W = time_frames // 4
        self.fc1 = nn.Linear(64 * H * W, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop — one full pass over the training DataLoader.
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"  loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


# ─────────────────────────────────────────────────────────────────────────────
# Test loop — evaluates accuracy and average loss on the held-out test set.
# ─────────────────────────────────────────────────────────────────────────────
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y      = X.to(device), y.to(device)
            pred      = model(X)
            test_loss += loss_fn(pred, y).item()
            correct   += (pred.argmax(1) == y).sum().item()

    print(f"  Accuracy: {100 * correct / size:.1f}%   "
          f"Avg loss: {test_loss / num_batches:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Whole-file inference using soft voting across segments.
#
# Steps:
#   1. Load the full audio file.
#   2. Slice into non-overlapping `duration`-second segments.
#   3. Run each segment through the model → softmax probabilities.
#   4. Average the probability vectors across all segments (soft vote).
#   5. The class with the highest average probability is the final prediction.
#
# Soft voting weights confident segments more heavily than uncertain ones,
# making it more robust than simple majority (hard) voting.
# ─────────────────────────────────────────────────────────────────────────────
def predict_file(fpath, model, sample_rate, duration, n_mfcc, categories, device):
    model.eval()
    audio, _  = librosa.load(fpath, sr=sample_rate)
    segments  = slice_into_segments(audio, sample_rate, duration)

    print(f"\n  File   : {os.path.basename(fpath)}")
    print(f"  Length : {len(audio)/sample_rate:.1f}s  →  {len(segments)} segment(s) of {duration}s\n")

    seg_probs = []
    with torch.no_grad():
        for i, seg in enumerate(segments):
            mfcc   = segment_to_mfcc(seg, sample_rate, n_mfcc).unsqueeze(0).to(device)
            logits = model(mfcc)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            seg_probs.append(probs)

            seg_label = categories[int(np.argmax(probs))]
            print(f"    Segment {i+1:>2d}: {seg_label:<16s}  (confidence {100*probs.max():.1f}%)")

    avg_probs   = np.mean(seg_probs, axis=0)
    final_idx   = int(np.argmax(avg_probs))
    final_label = categories[final_idx]

    print(f"\n  ── Soft-vote result {'─'*24}")
    for i, (cat, p) in enumerate(zip(categories, avg_probs)):
        bar    = "█" * int(p * 30)
        marker = " ◄" if i == final_idx else ""
        print(f"    {cat:<16s} {100*p:5.1f}%  {bar}{marker}")
    print(f"\n  FINAL PREDICTION : {final_label}  "
          f"(avg confidence {100*avg_probs[final_idx]:.1f}%)")
    print(f"  {'─'*44}\n")

    return final_label, avg_probs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Paths ─────────────────────────────────────────────────────────────────
    AUDIO_DIR     = r"C:\Users\owenw\OneDrive - purdue.edu\Desktop\respiratory-detector\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
    DIAGNOSIS_CSV = r"C:\Users\owenw\OneDrive - purdue.edu\Desktop\respiratory-detector\Respiratory_Sound_Database\Respiratory_Sound_Database\patient_diagnosis.csv"
    ASTHMA_DIR    = r"C:\Users\owenw\OneDrive - purdue.edu\Desktop\respiratory-detector\Asthma Detection Dataset Version 2\Asthma Detection Dataset Version 2"

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    SAMPLE_RATE   = 22050
    DURATION      = 3          # seconds per segment
    N_MFCC        = 40
    BATCH_SIZE    = 64
    EPOCHS        = 10
    LEARNING_RATE = 1e-3

    # Only these four diagnoses are used; all other labels are ignored
    CATEGORIES = ['COPD', 'Healthy', 'Pneumonia', 'Asthma']
    label2idx  = {label: idx for idx, label in enumerate(CATEGORIES)}

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Build datasets ────────────────────────────────────────────────────────
    print("Loading original dataset (CSV-labelled)...")
    dataset1 = AudioDataset(AUDIO_DIR, DIAGNOSIS_CSV, SAMPLE_RATE, DURATION, N_MFCC, label2idx)
    print(f"  {len(dataset1)} segments from {len(dataset1.file_samples)} files\n")

    print("Loading folder-based dataset...")
    dataset2 = FolderAudioDataset(ASTHMA_DIR, SAMPLE_RATE, DURATION, N_MFCC, label2idx)
    print(f"  {len(dataset2)} segments from {len(dataset2.file_samples)} files\n")

    # Merge into one combined dataset
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    dataset.file_samples = dataset1.file_samples + dataset2.file_samples
    print(f"Combined dataset: {len(dataset)} segments from "
          f"{len(dataset.file_samples)} files\n")

    # ── Audio preview (optional — close the plot window to continue) ──────────
    preview_random_sample(dataset, CATEGORIES, SAMPLE_RATE)

    # ── Train / test split ────────────────────────────────────────────────────
    train_size    = int(0.8 * len(dataset))
    test_size     = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    # num_workers=0 is required on Windows; increase to 2-4 on Linux/macOS
    train_dataloader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(
        test_data, batch_size=BATCH_SIZE, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    time_frames = (SAMPLE_RATE * DURATION) // 512 + 1
    model       = NeuralNet(
        num_classes=len(CATEGORIES), n_mfcc=N_MFCC, time_frames=time_frames
    ).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}  {'─'*30}")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader,   model, loss_fn,            device)
    print("Training complete!\n")

    # ── Whole-file soft-vote inference demo ───────────────────────────────────
    demo_fpath, true_label = random.choice(dataset.file_samples)
    print("=" * 45)
    print("🔬  Whole-file Soft-Vote Inference Demo")
    print("=" * 45)
    print(f"  True label : {CATEGORIES[true_label]}")
    predict_file(demo_fpath, model, SAMPLE_RATE, DURATION, N_MFCC, CATEGORIES, device)


if __name__ == "__main__":
    main()
