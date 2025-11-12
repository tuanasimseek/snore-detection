import tensorflow as tf

# 🔄 .h5 modelini yükle
model = tf.keras.models.load_model("horlama_model.keras")

# 🪄 TFLite dönüştürücüyü oluştur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔧 (İsteğe bağlı) Optimize et – boyutu küçültür
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 🔁 Dönüştür
tflite_model = converter.convert()

# 💾 Kaydet
with open("horlama_modeli.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ .tflite modeli başarıyla oluşturuldu!")
