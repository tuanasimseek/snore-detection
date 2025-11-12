def convert_to_cpp_array(model_filename, output_filename):
    # .tflite dosyasını açın ve okuma modunda açın
    with open(model_filename, "rb") as f:
        model_data = f.read()  # Modeli byte formatında okuyun

    # Model verisini C++ formatına dönüştürün
    cpp_array = ', '.join(f'0x{byte:02x}' for byte in model_data)  # Her byte'ı hex formatına çevirin
    cpp_code = f'const uint8_t model_data[] = {{ {cpp_array} }};'  # C++ array formatında veri oluşturun

    # C++ kodunu çıktı dosyasına yazın
    with open(output_filename, "w") as output_file:
        output_file.write(cpp_code)

    print(f"Model verisi {output_filename} dosyasına yazıldı.")  # Başarı mesajı

# Kullanım örneği
convert_to_cpp_array('horlama_modeli.tflite', 'horlama_modeli.cc')
