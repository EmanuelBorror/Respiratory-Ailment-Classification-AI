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
import time


## Define file and folder paths and parameters for configurations

AUDIO_DIR = r"C:\Users\emanu\OneDrive - purdue.edu\BME450\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
DIAGNOSIS_CSV = r"C:\Users\emanu\OneDrive - purdue.edu\BME450\Respiratory_Sound_Database\Respiratory_Sound_Database\patient_diagnosis.csv"

SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

CATEGORIES = [
    'COPD', 'Healthy', 'URTI', 'Bronchiectasis',
    'Pneumonia', 'Bronchiolitis', 'Asthma', 'LRTI'
]
label2idx = {label: idx for idx, label in enumerate(CATEGORIES)}

SAMPLE_NUM = 164

def preview_random_sample(dataset, categories):
    idx = random.randint(0, len(dataset.samples) - 1)
    fpath, label = dataset.samples[idx]

    print("=" * 45)
    print("🎧 Random Audio Sample Preview")
    print("=" * 45)
    print(f"  Index     : {idx}")
    print(f"  Label #   : {label}")
    print(f"  Category  : {categories[label]}")
    print(f"  File      : {os.path.basename(fpath)}\n")

    audio, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=DURATION)

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

preview_random_sample(AUDIO_DIR, CATEGORIES)