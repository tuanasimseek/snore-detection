#include <TensorFlowLite.h>
#include <ESP32Servo.h>
#include <U8g2lib.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <stdint.h>


// Model dosyası (başka bir .cc dosyasından gelmeli)
#include "horlama_modeli.cc"

// Donanım tanımlamaları
#define LED_PIN 2
#define MIC_PIN 35
#define SERVO1_PIN 13
#define SERVO2_PIN 12
#define I2S_NUM I2S_NUM_0
#define I2S_SAMPLESIZE 1024

// TensorFlow Lite ayarları
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Model
const tflite::Model* model = tflite::GetModel(g_model);

// Servo motorlar
Servo servo1;
Servo servo2;

// OLED ekran
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

// Eşik değeri
#define threshold 2000

// I2S mikrofon kurulumu
void i2s_init() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = I2S_SAMPLESIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = 26,
    .ws_io_num = 25,
    .data_out_num = -1,
    .data_in_num = 33
  };

  i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM, &pin_config);
  i2s_start(I2S_NUM);
}

void moveServos() {
  servo1.write(0);
  servo2.write(180);
  delay(1000);
  servo1.write(90);
  servo2.write(90);
  delay(1000);
}

void blinkLED(int seconds) {
  for (int i = 0; i < seconds * 2; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  
  // OLED
  Wire.begin(21, 22);
  u8g2.begin();
  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.clearBuffer();
  u8g2.setCursor(0, 20);
  u8g2.print("Sistem baslatiliyor...");
  u8g2.sendBuffer();

  // Servo
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo1.write(90);
  servo2.write(90);

  // Mikrofon
  i2s_init();

  // TFLM modeli başlat
  tflite::AllOpsResolver resolver;
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensorlar ayrilamadi!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model basariyla yuklendi!");

  u8g2.clearBuffer();
  u8g2.setCursor(0, 20);
  u8g2.print("Sistem hazir!");
  u8g2.sendBuffer();
  delay(1000);
}

void loop() {
  int micValue = analogRead(MIC_PIN);
  Serial.print("Ses seviyesi: ");
  Serial.println(micValue);

  u8g2.clearBuffer();
  u8g2.setCursor(0, 15);
  u8g2.print("Ses: ");
  u8g2.print(micValue);

  if (micValue > threshold) {
    u8g2.setCursor(0, 30);
    u8g2.print("Ses algilandi!");
    u8g2.sendBuffer();

    int16_t sample_buffer[I2S_SAMPLESIZE];
    size_t bytes_read;
    i2s_read(I2S_NUM, sample_buffer, sizeof(sample_buffer), &bytes_read, portMAX_DELAY);

    for (int i = 0; i < I2S_SAMPLESIZE && i < input->bytes / sizeof(float); i++) {
      input->data.f[i] = sample_buffer[i] / 32768.0f;
    }

    interpreter->Invoke();
    float* results = output->data.f;

    Serial.println("Sonuclar:");
    for (int i = 0; i < output->dims->data[1]; i++) {
      Serial.print("Kategori ");
      Serial.print(i);
      Serial.print(": ");
      Serial.println(results[i], 4);
    }

    if (results[0] > 0.6) {
      Serial.println("Horlama tespit edildi!");
      u8g2.clearBuffer();
      u8g2.setCursor(0, 20);
      u8g2.print("Horlama tespit!");
      u8g2.setCursor(0, 40);
      u8g2.print("Guven: ");
      u8g2.print(results[0], 2);
      u8g2.sendBuffer();

      blinkLED(2);
      moveServos();
    } else {
      u8g2.setCursor(0, 40);
      u8g2.print("Horlama degil");
      u8g2.sendBuffer();
    }
  } else {
    u8g2.setCursor(0, 30);
    u8g2.print("Dinleniyor...");
    u8g2.sendBuffer();
  }

  delay(200);
}