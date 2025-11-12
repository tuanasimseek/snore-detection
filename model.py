import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# 🔍 Veri yolu
veri_yolu = r"C:\Users\Lenovo\Desktop\veriler"

X = []
y = []

# 🔄 Ses dosyalarını oku ve MFCC çıkar
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

# 🔢 NumPy dizilerine çevir
X = np.array(X)
y = np.array(y)

# 🔀 Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Modeli oluştur ve eğit
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 🔍 Tahmin yap ve doğruluk hesapla
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Doğruluk oranı: {accuracy * 100:.2f}%")

# 💾 Modeli kaydet
joblib.dump(model, "horlama_modeli.pkl")

# Ağırlıkları ve bias değerlerini al
weights = model.coefs_
biases = model.intercepts_

# Örnek: 2 katmanlıysa:
print("1. Katman Ağırlıklar:", weights[0].shape)
print("1. Katman Bias:", biases[0].shape)

print("2. Katman Ağırlıklar:", weights[1].shape)
print("2. Katman Bias:", biases[1].shape)
