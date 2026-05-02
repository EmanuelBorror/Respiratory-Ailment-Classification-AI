# Fix for OpenMP conflict between PyTorch and librosa
import os
from sre_parse import CATEGORIES
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
import time


# Change the function signature to accept them
def preview_random_sample(dataset, categories, sample_rate, duration):
    idx = random.randint(0, len(dataset) - 1)
    fpath, label = dataset.samples[idx]

    print("=" * 45)
    print("🎧 Random Audio Sample Preview")
    print("=" * 45)
    print(f"  Index     : {idx}")
    print(f"  Label #   : {label}")
    print(f"  Category  : {categories[label]}")
    print(f"  File      : {os.path.basename(fpath)}\n")

    audio, sr = librosa.load(fpath, sr=sample_rate, duration=duration)  # use the parameters

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

# ____________________________________________________
# Define the neural network architecture for classification of .wav files 

class AudioDataset(Dataset):
    def __init__(self, audio_dir, diagnosis_csv, sample_rate, duration, n_mfcc, label2idx):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.label2idx = label2idx

        df = pd.read_csv(diagnosis_csv, header=None, names=['patient_id', 'diagnosis'])
        self.patient_labels = dict(zip(df['patient_id'].astype(str), df['diagnosis']))

        self.samples = []  # list of (filepath, label_idx)
        for fname in os.listdir(audio_dir):
            if fname.endswith('.wav'):
                patient_id = fname.split('_')[0]  # files are named like 101_1b1_Al_sc_Meditron.wav
                diagnosis = self.patient_labels.get(patient_id)
                if diagnosis in label2idx:
                    self.samples.append((os.path.join(audio_dir, fname), label2idx[diagnosis]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        audio, _ = librosa.load(fpath, sr=self.sample_rate, duration=self.duration)

        # Pad or truncate to fixed length
        target_len = self.sample_rate * self.duration
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # shape: (1, n_mfcc, time)
        return mfcc, label

class NeuralNet(nn.Module):
    def __init__(self, num_classes, n_mfcc, time_frames):
        super(NeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU (),
            nn.AdaptiveAvgPool2d((1, None))          
        )

        self.rnn = nn.GRU(input_size = 128, hidden_size = 64, batch_first = True, bidirectional = True)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(64, num_classes)
        )
        # H = n_mfcc // 4          # two MaxPool2d halvings
        # W = time_frames // 4
        # self.fc1 = nn.Linear(64 * H * W, 128)  # integer math, not float /
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        _, h_n = self.rnn(x)
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.classifier(x)   

        # x = self.pool(F.relu(self.conv1(x)))   # use imported F, not nn.functional inline
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))                # missing relu + fc1 call in original
        # x = F.relu(self.fc2(x))                # missing relu + fc2 call in original
        # x = F.relu(self.fc3(x))                # missing relu + fc3 call
        # x = F.relu(self.fc4(x))                # missing relu + fc4 call
        # return self.fc5(x)
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    AUDIO_DIR = r"C:\Users\emanu\OneDrive - purdue.edu\BME450\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
    DIAGNOSIS_CSV = r"C:\Users\emanu\OneDrive - purdue.edu\BME450\Respiratory_Sound_Database\Respiratory_Sound_Database\patient_diagnosis.csv"

    SAMPLE_RATE = 22050
    DURATION = 5
    N_MFCC = 40
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    CATEGORIES = ['COPD', 'Healthy', 'URTI', 'Bronchiectasis',
                  'Pneumonia', 'Bronchiolitis', 'Asthma', 'LRTI']
    label2idx = {label: idx for idx, label in enumerate(CATEGORIES)}

    dataset = AudioDataset(AUDIO_DIR, DIAGNOSIS_CSV, SAMPLE_RATE, DURATION, N_MFCC, label2idx)

    preview_random_sample(dataset, CATEGORIES, SAMPLE_RATE, DURATION)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Compute time_frames for NeuralNet sizing
    time_frames = (SAMPLE_RATE * DURATION) // 512 + 1

    model = NeuralNet(num_classes=len(CATEGORIES), n_mfcc=N_MFCC, time_frames=time_frames)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    sample_num = 143
    with torch.no_grad():
        r = model(training_data[sample_num][0].unsqueeze(0))  # needs batch dim

    print('output pseudo-probabilities:', r)
    print('predicted class number:', torch.argmax(r).item())
    print('predicted class:', CATEGORIES[torch.argmax(r).item()])

if __name__ == "__main__":
    main()
