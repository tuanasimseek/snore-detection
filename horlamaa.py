import numpy as np
import time

# Simülasyon için (gerçek ses yok)
def read_microphone():
    # Ses şiddetini rastgele üret (örneğin 0 - 4000 arası)
    return np.random.randint(1000, 4000)

# Simülasyon için (gerçek model yok)
def load_model():
    print("Model yüklendi!")
    return None  # Burada gerçek bir model yerine simülasyon var

def run_inference(model, input_data):
    # Bu kısımda gerçek bir model çalıştırılabilir
    # Simülasyon: horlama varsa 0.7, yoksa 0.2
    return [0.7 if input_data > 2500 else 0.2]

def main():
    model = load_model()
    threshold = 2000

    while True:
        mic_value = read_microphone()
        print(f"Ses seviyesi: {mic_value}")

        if mic_value > threshold:
            print("Ses algılandı. Model çalıştırılıyor...")
            result = run_inference(model, mic_value)
            print(f"Model sonucu: {result[0]:.2f}")

            if result[0] > 0.6:
                print("Horlama tespit edildi! LED ve Servo tetikleniyor.")
            else:
                print("Horlama değil.")
        else:
            print("Dinleniyor...")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
