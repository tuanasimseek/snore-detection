# Horlama Tespitli 

Bu proje, horlama sesini makine öğrenmesi modeliyle algılayan ve yastığın içerisindeki titreşim motorlarını çalıştıran Arduino tabanlı bir sistemdir.

## Projenin Amacı

Projenin amacı, kullanıcı horlamaya başladığında bunu otomatik olarak tespit etmek ve dört adet titreşim motorunu çalıştırarak kullanıcının uyku pozisyonunu değiştirmesine yardımcı olmaktır.

## Nasıl Çalışır?

1. Mikrofon ortamdan ses verisi toplar.
2. Eğitilmiş makine öğrenmesi modeli sesi analiz eder.
3. Horlama sesi tespit edildiğinde Arduino'ya sinyal gönderilir.
4. Yastığın içerisine yerleştirilen dört titreşim motoru çalıştırılır.
5. Titreşim sayesinde kullanıcının uyku pozisyonunu değiştirmesi hedeflenir.

## Kullanılan Malzemeler

- Arduino veya TensorFlow Lite destekleyen bir mikrodenetleyici
- Mikrofon veya ses sensörü
- 4 adet titreşim motoru
- Motor sürücü devresi veya transistör/MOSFET
- Uygun güç kaynağı
- Bağlantı kabloları
- Yastık
- Bilgisayar

> Titreşim motorları Arduino pinlerine doğrudan bağlanmamalıdır. Motor sürücü devresi veya uygun bir transistör/MOSFET kullanılmalıdır.

## Kullanılan Teknolojiler

- **Python:** Modelin eğitilmesi
- **Scikit-learn / TensorFlow Lite:** Ses sınıflandırma modeli
- **C++ / Arduino:** Donanım kontrolü
- **Arduino IDE:** Kodun mikrodenetleyiciye yüklenmesi

## Modelin Çalışma Süreci

Horlama ve normal ortam seslerinden oluşan ses verileri kullanılarak bir sınıflandırma modeli eğitilmiştir. Eğitilen model, mikrodenetleyicide çalışabilecek formata dönüştürülmüş ve Arduino koduna eklenmiştir.

Model horlama sesi algıladığında çıkış pinleri aktif hale gelir ve yastığın içerisindeki dört titreşim motoru aynı anda çalışır.

## Dosya Yapısı

- `egit.py` — Model eğitim işlemleri
- `horlama_model.py` — Model mimarisi
- `convert_model.py` — Model dönüştürme işlemleri
- `horlama_modeli.cc` — Mikrodenetleyiciye gömülen model
- `horlama.ino` — Arduino ve titreşim motoru kontrol kodu

## Kurulum

1. Gerekli donanım bağlantılarını yapın.
2. Mikrofonu ve dört titreşim motorunu Arduino'ya bağlayın.
3. Motorları uygun bir sürücü devresi üzerinden kontrol edin.
4. `horlama.ino` dosyasını Arduino IDE ile açın.
5. Kart ve port seçimini yapın.
6. Kodu mikrodenetleyiciye yükleyin.
7. Mikrofonu ve titreşim motorlarını yastığın içerisine uygun şekilde yerleştirin.
8. Sistemi çalıştırarak horlama sesiyle test edin.

## Proje Akışı

```text
Mikrofon
   ↓
Ses Verisinin Alınması
   ↓
Makine Öğrenmesi Modeli
   ↓
Horlama Tespiti
   ↓
Arduino
   ↓
4 Titreşim Motorunun Çalışması
