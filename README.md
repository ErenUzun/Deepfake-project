# Machine Learning ile Yapay Zeka Görüntü Sahteciliği Tespiti Projesi 🛡️

Bu proje; üretici yapay zeka, GAN ve latent difüzyon modelleri (Stable Diffusion v1.4) tarafından sentezlenen sentetik görselleri, pikselsel düzeydeki frekans anomalilerini ve jeneratör izlerini analiz ederek tespit eden uçtan uca derin öğrenme tabanlı bir siber güvenlik/dijital adli tıp çözümüdür.

## 🚀 Öne Çıkan Mühendislik Özellikleri
- **Hafif Custom CNN Mimarisi:** Bilgisayar donanım kaynaklarını optimize eden, ardışık 4 evrişim bloğundan oluşan ve aşırı öğrenmeyi engelleyen (%50 Dropout & BatchNorm) özel sinir ağı tasarımı.
- **Explainable AI (XAI) Entegrasyonu:** Derin öğrenme modelinin "Kara Kutu" (Black Box) handikapını kıran ve kararın pikselsel gerekçelerini OpenCV JET renk paletiyle görselleştiren Vanilla Saliency Map motoru.
- **Asenkron Web UI:** Python tabanlı Streamlit framework'ü ile tasarlanmış; anlık tahmin, güven oranı kartları ve ısı haritası analizini saniyenin onda birinden kısa sürede sunabilen dinamik gösterge paneli.

## 📊 Deneysel Bulgular ve Performans Metrikleri
Bilgisayarla görme literatüründe akademik bir standart (benchmark) kabul edilen **CIFAKE Veri Seti** (120.000 RGB imaj, 32x32 yerel çözünürlük) üzerinde NVIDIA RTX 4060 Ti GPU donanımı kullanılarak yürütülen 10 Epochluk eğitim neticesinde şu başarılara ulaşılmıştır:

- **Sınıflandırma Doğruluğu (Test Accuracy):** %95.23
- **F1-Skoru (F1-Score):** %95.22
- **Eğitim Kontrolü:** Model, veri ön işleme hattında Bilinear Interpolation ile yapay olarak 224x224 boyutuna upsampling yapılan imajlar üzerinde aşırı öğrenme (overfitting) tuzağına düşmeden dengeli bir öğrenme eğrisi yakalamıştır.

## 📦 Proje Dizin Yapısı
Proje, nesne yönelimli programlama prensiplerine uygun, modüler ve sürdürülebilir bir mimaride kurgulanmıştır:
```text
├── models/
│   └── detector_model.pth    # Eğitilmiş ve serialize edilmiş en iyi model ağırlıkları
├── src/
│   ├── data_loader.py         # Bozuk imaj toleranslı ve ImageNet normalizasyonlu veri yükleyici
│   ├── model.py               # Custom CNN mimarisinin katman tanımlamaları
│   └── train.py               # Adam optimizasyonu ve Cross-Entropy tabanlı eğitim döngüsü
├── app.py                     # Streamlit tabanlı web arayüzü ve asenkron çıkarım köprüsü
├── debug_model.py             # Lokal test ve hata ayıklama scripti
└── .gitignore                 # Gereksiz önbellek ve yerel IDE dosyalarını filtreleyen zırh
```

🛠️ Kurulum ve Çalıştırma
1. Depoyu Klonlayın veya İndirin

Bash
git clone [https://github.com/ErenUzun/Deepfake-project.git](https://github.com/ErenUzun/Deepfake-project.git)
cd Deepfake-project

3. Bağımlılıkları Yükleyin
 
Bash
pip install torch torchvision numpy Pillow opencv-python streamlit matplotlib

5. Web Arayüzünü Başlatın

Bash
streamlit run app.py
