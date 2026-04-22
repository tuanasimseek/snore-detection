# Horlama Tespiti Sistemi (Snore Detection)

Ses analizi ve makine öğrenmesi kullanarak gerçek zamanlı horlama tespiti yapan,  
Arduino tabanlı IoT sistemi.

## Nasıl Çalışıyor?
1. Mikrofon ses verisini toplar
2. Python'da eğitilen ML modeli sesi analiz eder
3. Horlama tespit edilirse Arduino üzerinden titreşim motoru devreye girer
4. Model C++ formatına dönüştürülerek gömülü sisteme yüklenir

##  Teknolojiler
- **Python** — Model eğitimi (scikit-learn)
- **C++ / Arduino** — Gömülü sistem entegrasyonu
- **TFLite** — Modelin mikrodenetleyiciye taşınması

##  Dosya Yapısı
- `egit.py` — Model eğitimi
- `horlama_model.py` — Model mimarisi
- `convert_model.py` — TFLite dönüşümü
- `horlama_modeli.cc` — C++ gömülü model
- `horlama.ino` — Arduino kodu
