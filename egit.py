import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 🔍 Veri yolu
veri_yolu = r"C:\Users\Lenovo\Desktop\veriler"

X = []
y = []

# MFCC çıkar
for klasor_adi in os.listdir(veri_yolu):
    klasor_yolu = os.path.join(veri_yolu, klasor_adi)
    if os.path.isdir(klasor_yolu):
        for dosya in os.listdir(klasor_yolu):
            if dosya.endswith(".wav"):
                dosya_yolu = os.path.join(klasor_yolu, dosya)
                try:
                    ses, sr = librosa.load(dosya_yolu, sr=16000)
                    mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=13)
                    mfcc_ortalama = np.mean(mfcc.T, axis=0)
                    X.append(mfcc_ortalama)
                    y.append(1 if "horlama" in klasor_adi.lower() else 0)
                except Exception as e:
                    print(f"Hata oluştu: {dosya_yolu} – {e}")

X = np.array(X)
y = np.array(y)

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Keras model
model = keras.Sequential([
    layers.Input(shape=(13,)),  # MFCC boyutu
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Eğit
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Kaydet
model.save("horlama_model.keras")

