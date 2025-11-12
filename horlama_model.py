import librosa
import numpy as np
import os

def extract_features(directory, label):
    features = []
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            audio, sr = librosa.load(file_path, duration=3, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_scaled = np.mean(mfcc.T, axis=0)  # (13,) boyutunda
            features.append((mfcc_scaled, label))
    return features

horlama = extract_features("dataset/segment", 1)
normal = extract_features("dataset/normal", 0)

all_data = horlama + normal
X = np.array([x[0] for x in all_data])
y = np.array([x[1] for x in all_data])
