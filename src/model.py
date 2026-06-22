import torch
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    """
    Akademik Jüride Savunulabilir Hafif Custom CNN Mimarisi
    
    Katman Katman Boyut ve Filtre Değişim Tablosu:
    --------------------------------------------------------
    Giriş Görseli       : [Batch,   3, 224, 224]
    Blok 1 (Conv+Pool)  : [Batch,  16, 112, 112]
    Blok 2 (Conv+Pool)  : [Batch,  32,  56,  56]
    Blok 3 (Conv+Pool)  : [Batch,  64,  28,  28]
    Blok 4 (Conv+Pool)  : [Batch, 128,  14,  14]
    Düzleştirme (Flatten): [Batch, 25088] (128 * 14 * 14)
    Sınıflandırıcı Başlığı: [Batch, 2] (REAL=0, FAKE=1)
    --------------------------------------------------------
    """
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        # 1. Blok: 3 -> 16 filtre, Boyut: 224x224 -> MaxPool ile 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. Blok: 16 -> 32 filtre, Boyut: 112x112 -> MaxPool ile 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 3. Blok: 32 -> 64 filtre, Boyut: 56x56 -> MaxPool ile 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 4. Blok: 64 -> 128 filtre, Boyut: 28x28 -> MaxPool ile 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Düzleştirme katmanı
        self.flatten = nn.Flatten()
        
        # Sınıflandırıcı (Classifier Head) + Overfitting Koruması (Dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # Ezberlemeyi önlemek için nöronların %50'sini kapatır
            nn.Linear(128 * 14 * 14, 2) # 25088 giriş -> 2 çıkış (REAL/FAKE)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

# ============ SUB-STAGE 2.1: DRY-RUN KURU ÇALIŞTIRMA TESTİ ============
if __name__ == '__main__':
    print("Sub-Stage 2.1: Model Mimarisi Kuru Çalıştırma Testi Başlatılıyor...")
    
    try:
        # 1. Modeli Örnekle
        model = DeepfakeDetector()
        
        # 2. Data Loader'dan gelecekmiş gibi sahte bir Batch oluştur (Batch_size=1, 3 Kanal, 224x224 boyut)
        fake_image_batch = torch.randn(1, 3, 224, 224)
        print(f"  -> Giriş Tensor Boyutu: {fake_image_batch.shape} (Beklenen: [1, 3, 224, 224])")
        
        # 3. Bu sahte veriyi modelin içine besle (Forward Pass)
        output_logits = model(fake_image_batch)
        
        print("\n✓ Sub-Stage 2.1 Mimarisi: BAŞARILI!")
        print(f"  -> Model Çıktı Tensor Boyutu: {output_logits.shape} (Beklenen: [1, 2])")
        print("  -> Matematiksel Boyut Hesaplaması Kusursuz. Shape Mismatch Riski Yok.")
        
    except Exception as e:
        print(f"\n[X] Mimari Tasarım Hatası: {e}")