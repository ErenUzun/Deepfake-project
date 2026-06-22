import torch
import torch.nn as nn
from PIL import Image
from src.model import DeepfakeDetector
from torchvision import transforms
import os

print("=" * 70)
print("🔬 MODEL DEBUG - TAHMIN ORANLARINI KONTROL ET")
print("=" * 70)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}\n")

# Model yükle
print("[*] Model yükleniyor...")
model = DeepfakeDetector().to(device)
model_path = "models/detector_model.pth"

if not os.path.exists(model_path):
    print(f"❌ HATA: Model dosyası bulunamadı: {model_path}")
    exit(1)

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model başarıyla yüklendi\n")
except Exception as e:
    print(f"❌ HATA: Model yükleme hatası: {e}")
    exit(1)

# Transform (aynı app.py'de kullanılan)
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test fotoğrafı seç (FAKE olması gereken)
test_image_path = input("[?] Test fotoğraf yolunu gir (örn: data/test/FAKE/test.jpg): ").strip()

if not os.path.exists(test_image_path):
    print(f"❌ HATA: Dosya bulunamadı: {test_image_path}")
    exit(1)

print(f"\n[*] Fotoğraf yükleniyor: {test_image_path}")

try:
    image = Image.open(test_image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    print("✅ Fotoğraf başarıyla yüklendi ve transform uygulandı\n")
except Exception as e:
    print(f"❌ HATA: Fotoğraf yükleme hatası: {e}")
    exit(1)

# Model inference
print("[*] Model inference başlatılıyor...\n")

with torch.no_grad():
    raw_outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(raw_outputs, dim=1)

print("=" * 70)
print("📊 SONUÇLAR")
print("=" * 70)

# Raw outputs
print(f"\n[RAW OUTPUTS] (normalize öncesi)")
print(f"  Output shape: {raw_outputs.shape}")
print(f"  Class 0 (REAL): {raw_outputs[0][0].item():.6f}")
print(f"  Class 1 (FAKE): {raw_outputs[0][1].item():.6f}")

# Softmax outputs
print(f"\n[SOFTMAX OUTPUTS] (yüzdelik)")
real_prob = probabilities[0][0].item() * 100
fake_prob = probabilities[0][1].item() * 100
print(f"  Class 0 (REAL): {real_prob:.2f}%")
print(f"  Class 1 (FAKE): {fake_prob:.2f}%")

# Tahmin
predicted_class = torch.argmax(raw_outputs, dim=1).item()
class_names = {0: "REAL", 1: "FAKE"}
print(f"\n[TAHMİN] Model kararı: {class_names[predicted_class]}")

if predicted_class == 1:
    print("✅ DOĞRU: FAKE olarak sınıflandırıldı")
else:
    print("❌ YANLIŞ: REAL olarak sınıflandırıldı (BU HATA!)")

print("\n" + "=" * 70)