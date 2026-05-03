import os
from sre_parse import CATEGORIES
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
import sounddevice as sd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import pandas as pd

# ____________________________________________________
# Helper functions
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

def compute_features(chunk, sample_rate, n_mfcc):
    mfcc    = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=n_mfcc)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)   # (n_mfcc*3, T)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

def precompute_features(file_samples, sample_rate, duration, n_mfcc):
    segment_len = sample_rate * duration

    total_segments = 0
    for fpath, _ in file_samples:
        info = sf.info(fpath)
        total_segments += max(1, int(np.ceil(info.duration / duration)))

    print(f"    Total segments to pre-compute : {total_segments}")
    print(f"    Across {len(file_samples)} audio files\n")

    cache     = []
    seg_count = 0

    for fpath, label in file_samples:
        audio, _ = librosa.load(fpath, sr=sample_rate)

        if len(audio) == 0:
            silence = np.zeros(segment_len, dtype=np.float32)
            tensor  = compute_features(silence, sample_rate, n_mfcc)
            cache.append((tensor, label))
            seg_count += 1
            continue

        start = 0
        while start < len(audio):
            chunk = audio[start : start + segment_len]
            if len(chunk) < segment_len:
                chunk = np.pad(chunk, (0, segment_len - len(chunk)))

            tensor = compute_features(chunk, sample_rate, n_mfcc)
            cache.append((tensor, label))

            start     += segment_len
            seg_count += 1

            if seg_count % 100 == 0:
                pct = 100 * seg_count / total_segments
                print(f"    [{seg_count:>5d} / {total_segments}]  {pct:.1f}% complete")

    print(f"    Done — {seg_count} segments cached in RAM.\n")
    return cache, seg_count

def augment_features(tensor):
    """
    tensor shape: (1, n_features, time_frames)
    Returns an augmented copy — does NOT modify the cached original.
    """
    t = tensor.clone()
    _, n_freq, n_time = t.shape

    # Time masking — blank out up to 1/8 of the time axis
    if n_time > 8:
        t_mask  = random.randint(1, max(1, n_time // 8))
        t_start = random.randint(0, n_time - t_mask)
        t[:, :, t_start : t_start + t_mask] = 0.0

    # Frequency masking — blank out up to 1/8 of the frequency axis
    if n_freq > 8:
        f_mask  = random.randint(1, max(1, n_freq // 8))
        f_start = random.randint(0, n_freq - f_mask)
        t[:, f_start : f_start + f_mask, :] = 0.0

    return t

def preview_random_sample(dataset, categories, sample_rate, duration):
    idx = random.randint(0, len(dataset) - 1)
    fpath, label = dataset.samples[idx]

    print("=" * 45)
    print("🎧 Random Audio Sample Preview")
    print("=" * 45)
    print(f"  Index      : {idx}")
    print(f"  Label #    : {label}")
    print(f"  Category   : {categories[label]}")
    print(f"  File       : {os.path.basename(fpath)}\n")

    audio, sr = librosa.load(fpath, sr=sample_rate, duration=duration)

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

def plot_dataset_distribution(train_dataset, test_dataset, categories):
    # Get counts for each class in train and test sets
    train_counts = Counter([s[1] for s in train_dataset.samples])
    test_counts = Counter([s[1] for s in test_dataset.samples])
    
    # Create a DataFrame for easy plotting
    data = {
        'Category': categories,
        'Train': [train_counts.get(i, 0) for i in range(len(categories))],
        'Test': [test_counts.get(i, 0) for i in range(len(categories))]
    }
    df = pd.DataFrame(data).set_index('Category')
    save_option = input("Do you want to save the dataset distribution plot? (y/n): ").strip().lower()
    # Plotting
    df.plot(kind='bar', figsize=(10, 6), color=['#4C72B0', '#55A868'])
    plt.title('Distribution of Samples: Train vs Test Sets')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_option == 'y':
        plt.savefig('dataset_distribution.png')
        print("Dataset distribution plot saved as 'dataset_distribution.png'")
    else: 
        print("Dataset distribution plot not saved.")

    plt.show()

# ____________________________________________________
# Dataset and Model Definition

# Focal Loss Implementation to handle class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# class AudioDataset(Dataset):
#     def __init__(self, audio_dir, sample_rate, duration, n_mfcc, label2idx):
#         self.sample_rate = sample_rate
#         self.duration = duration
#         self.n_mfcc = n_mfcc
#         self.samples = []

#         # Iterate through directories to find .wav files
#         for label_name in os.listdir(audio_dir):
#             label_dir = os.path.join(audio_dir, label_name)
#             if os.path.isdir(label_dir) and label_name in label2idx:
#                 for fname in os.listdir(label_dir):
#                     if fname.endswith('.wav'):
#                         fpath = os.path.join(label_dir, fname)
#                         self.samples.append((fpath, label2idx[label_name]))

#     def __len__(self): 
#         return len(self.samples)

#     def __getitem__(self, idx):
#         fpath, label = self.samples[idx]
#         # Load audio, ensuring it matches the duration defined in main
#         audio, _ = librosa.load(fpath, sr=self.sample_rate, duration=self.duration)
        
#         target_len = self.sample_rate * self.duration
#         audio = np.pad(audio, (0, max(0, target_len - len(audio))))[:target_len]
        
#         mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
#         return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label

class AudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate, duration, n_mfcc, label2idx):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.samples = []  # This will now store (mfcc_tensor, label)

        segment_len = int(sample_rate * duration)

        print(f"Processing audio files from {audio_dir}...")
        for label_name in os.listdir(audio_dir):
            label_dir = os.path.join(audio_dir, label_name)
            if os.path.isdir(label_dir) and label_name in label2idx:
                label_idx = label2idx[label_name]
                for fname in os.listdir(label_dir):
                    if fname.endswith('.wav'):
                        fpath = os.path.join(label_dir, fname)
                        
                        # Load full audio
                        audio, _ = librosa.load(fpath, sr=sample_rate)
                        
                        # Slice into segments
                        start = 0
                        while start + segment_len <= len(audio):
                            segment = audio[start : start + segment_len]
                            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
                            self.samples.append((torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label_idx))
                            start += segment_len
                        
                        # Optional: Include the final fragment if it's shorter than duration
                        # by padding it (ensures no data is lost)
                        if start < len(audio):
                            remaining = audio[start:]
                            padding = segment_len - len(remaining)
                            segment = np.pad(remaining, (0, padding))
                            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
                            self.samples.append((torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label_idx))

        # print(f"Dataset ready with {len(self.samples)} segments.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels), nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels)
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.conv(x) + self.shortcut(x))

# class NeuralNet(nn.Module): # CNN + BiGRU + Attention
#     def __init__(self, num_classes):
#         super(NeuralNet, self).__init__()
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(1, 32, kernel_size=3, padding=1), 
#             nn.BatchNorm2d(32), nn.ReLU(), 
#             nn.MaxPool2d(2), nn.Dropout2d(0.2),
            
#             # Block 2
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), 
#             nn.BatchNorm2d(64), nn.ReLU(), 
#             nn.MaxPool2d(2), nn.Dropout2d(0.2),
            
#             # Block 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), 
#             nn.BatchNorm2d(128), nn.ReLU(), 
#             nn.AdaptiveAvgPool2d((1, None)), nn.Dropout2d(0.2)
#         )
#         self.rnn = nn.GRU(128, 64, batch_first=True, bidirectional=True)
#         self.attention = Attention(64)
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 64), nn.ReLU(), 
#             nn.Dropout(0.4), 
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x).squeeze(2).permute(0, 2, 1)
#         x, _ = self.rnn(x)
#         x = self.attention(x)
#         # Concatenate final hidden states from both directions of GRU
#         # torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
#         return self.classifier(x)

class NeuralNet(nn.Module): # CNN + BiGRU
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, None))
        )
        self.rnn = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x).squeeze(2).permute(0, 2, 1)
        _, h_n = self.rnn(x)
        return self.classifier(torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1))
    
# class NeuralNet(nn.Module): # CNN + Transformer
#     def __init__(self, num_classes):
#         super(NeuralNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, None)) 
#         )
#         self.proj = nn.Linear(128, 64)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.classifier = nn.Sequential(
#             nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.squeeze(2).permute(0, 2, 1)
#         x = self.proj(x)
#         x = self.transformer(x)
#         x = x.mean(dim=1)
#         return self.classifier(x)

# ____________________________________________________
# Training and Testing Loops

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_loop(dataloader, model, loss_fn, categories):
    model.eval()
    all_preds, all_labels = [], []
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_test_loss = test_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAvg Test Loss: {avg_test_loss:.4f}")
    print("\nPer-Category Classification Report:")
    print(classification_report(all_labels, all_preds, labels=list(range(len(categories))), 
                                target_names=categories, zero_division=0))
    return all_labels, all_preds, avg_test_loss, accuracy

def get_weighted_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    num_samples = len(targets)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

# ____________________________________________________
# Main Execution

def main():
    train_dir = r'C:\Users\emanu\OneDrive - purdue.edu\Documents\PyTorch Scripts.Me\BME450 Project\AudioDataset\Asthma Detection Dataset Version 2\Train'
    test_dir = r'C:\Users\emanu\OneDrive - purdue.edu\Documents\PyTorch Scripts.Me\BME450 Project\AudioDataset\Asthma Detection Dataset Version 2\Test'

    num_epochs = 10
    
    SAMPLE_RATE, DURATION, N_MFCC = 22050, 5, 40
    CATEGORIES = ['asthma', 'Bronchial', 'copd', 'healthy', 'pneumonia']
    label2idx = {label: idx for idx, label in enumerate(CATEGORIES)}
    
    train_data = AudioDataset(train_dir, SAMPLE_RATE, DURATION, N_MFCC, label2idx)
    test_data = AudioDataset(test_dir, SAMPLE_RATE, DURATION, N_MFCC, label2idx)
    
    # Preview sample before starting
    # preview_random_sample(train_data, CATEGORIES, SAMPLE_RATE, DURATION)
    plot_dataset_distribution(train_data, test_data, CATEGORIES)

    sampler = get_weighted_sampler(train_data)
    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=32)
    # time_frames = (SAMPLE_RATE * DURATION) // 512 + 1

    model = NeuralNet(len(CATEGORIES))
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    final_labels, final_preds = [], []
    train_history = []
    test_history = []
    accuracy_history = []
    
    for t in range(num_epochs):
        print(f"\nEpoch {t+1}")
        # Track training loss
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        train_history.append(train_loss)
        print(f"Epoch {t+1} Training Loss: {train_loss:.4f}")
        # Track testing loss
        final_labels, final_preds, test_loss, accuracy = test_loop(test_loader, model, loss_fn, CATEGORIES)
        test_history.append(test_loss)
        accuracy_history.append(accuracy)

    save_option = input("Do you want to save the Loss Curve? (y/n): ").strip().lower()

    if save_option == 'y':
        file_name = input("Enter desired file name: ").strip()
        if not file_name.endswith('.png'):
            file_name += '.png'
            # Plot Loss Curve
            plt.figure(figsize=(10, 5))
            plt.plot(train_history, label='Train Loss')
            plt.plot(test_history, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Testing Loss per Epoch')
            plt.legend()
            plt.savefig(file_name, dpi=300)
            plt.show()
            print(f"Loss curve saved as {file_name}")
    else: 
        # Plot Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_history, label='Train Loss')
        plt.plot(test_history, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss per Epoch')
        plt.legend()
        plt.show()
        print("Loss curve not saved.")

    save_option = input("Do you want to save the Accuracy Curve? (y/n): ").strip().lower()

    if save_option == 'y':
        file_name = input("Enter desired file name: ").strip()
        if not file_name.endswith('.png'):
            file_name += '.png'
            # Plot Accuracy Curve
            plt.figure(figsize=(10, 5))
            plt.plot(accuracy_history, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Testing Accuracy per Epoch')
            plt.legend()
            plt.savefig(file_name, dpi=300)
            plt.show()
            print(f"Accuracy curve saved as {file_name}")
    else: 
        # Plot Accuracy Curve
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_history, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Testing Accuracy per Epoch')
        plt.legend()
        plt.show()
        print("Accuracy curve not saved.")

    save_option = input("Do you want to save the confusion matrix? (y/n): ").strip().lower()

    if save_option == 'y':
        file_name = input("Enter desired file name: ").strip()
        if not file_name.endswith('.png'):
            file_name += '.png'
            cm = confusion_matrix(final_labels, final_preds, labels=list(range(len(CATEGORIES))))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=CATEGORIES, yticklabels=CATEGORIES)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Bias Chart)')
            plt.savefig(file_name, dpi=300)
            plt.show()
            print(f"Confusion matrix saved as {file_name}")
    else: 
        cm = confusion_matrix(final_labels, final_preds, labels=list(range(len(CATEGORIES))))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=CATEGORIES, yticklabels=CATEGORIES)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Bias Chart)')
        plt.show()
        print("Confusion matrix not saved.")   

if __name__ == "__main__":
    main()