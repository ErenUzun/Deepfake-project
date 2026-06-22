import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import logging
import os

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ GLOBAL SEED = 42 ============
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Tolerant image loading for truncated or partially corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============ CUSTOM TRANSFORM: GIF & RGB HANDLING ============
class SafeImageTransform:
    """
    Görselleri modelin istediği formata güvenli şekilde getiren transform sınıfı:
    1. Animasyonlu GIF gelirse sadece ilk kareyi alır.
    2. RGB olmayan (Gri ton, RGBA vb.) görselleri 3 kanallı RGB'ye çevirir.
    3. Resimleri Bilinear Interpolation ile 224x224 boyutuna büyütür.
    4. Model uyumu için ImageNet ortalama ve standart sapma değerleriyle normalize eder.
    """
    def __init__(self, size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.size = size
        self.normalize = transforms.Normalize(mean=mean, std=std)
    
    def __call__(self, img):
        # GIF Safety: Extract the first frame only
        if hasattr(img, 'is_animated') and img.is_animated:
            img.seek(0)
        
        # Color Space Safety: Force RGB conversion
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Geometry Safety: Resize with Bilinear Interpolation to 224x224
        img = img.resize((self.size, self.size), Image.BILINEAR)
        
        # Convert to Tensor & Apply Academic ImageNet Normalization
        img_tensor = transforms.ToTensor()(img)
        img_tensor = self.normalize(img_tensor)
        
        return img_tensor

# Instantiate the structured transform pipeline
transform = SafeImageTransform()

# ============ SAFE DATASET WRAPPER ============
class SafeImageFolder(datasets.ImageFolder):
    """
    Klasörden veri okurken bozuk bir dosyaya denk gelirse training sürecinin
    yarıda kesilmesini engeller. Hatayı loga yazar ve bir sonraki resme geçer
    """
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except Exception as e:
            path, _ = self.samples[index]
            logger.warning(f"Skipping corrupted image: {path}. Error: {e}")
            
            # Cyclic safe recovery: try loading the next index
            next_index = (index + 1) % len(self)
            return self.__getitem__(next_index)

# ============ DATALOADER FACTORY FUNCTION ============
def get_data_loaders(data_dir='./data', batch_size=32, num_workers=4):
    """
    Train ve Test klasörlerindeki verileri PyTorch DataLoader modeline bağlar.
    - Klasör sıralamasına göre otomatik etiketleme: REAL -> 0, FAKE -> 1 olur.
    - Performans için pin_memory aktiftir.
    """
    
    train_dataset = SafeImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
    )
    
    test_dataset = SafeImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training set for optimal learning convergence
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # Keeping evaluation sequential and steady
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ============ GÖZLE KONTROL VE DOĞRULAMA TESTİ ============
if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] Hata: Matplotlib kütüphanesi bulunamadı. Lütfen terminale 'pip install matplotlib' yazıp tekrar dene.")
        exit(1)
        
    print("DataLoader ayağa kaldırılıyor ve ilk batch çekiliyor (Lütfen bekleyin)...")
    
    try:
        # Test için batch_size=4 yapıyoruz ki yan yana 4 resim görelim
        train_loader, test_loader = get_data_loaders(batch_size=4) 
        
        # İlk 4 resmi ve etiketini RAM'den yakala
        images, labels = next(iter(train_loader))
        
        print("\n Stage 1 Data Pipeline: BAŞARILI!")
        print(f"  -> Toplam Train Batch Sayısı: {len(train_loader)}")
        print(f"  -> Toplam Test Batch Sayısı: {len(test_loader)}")
        print(f"  -> Çekilen Batch Tensor Boyutu: {images.shape} (Beklenen: [4, 3, 224, 224])")
        print("  -> Etiket Kuralı: REAL=0, FAKE=1\n")
        print(" Resimler ekrana basılıyor... Lütfen açılan pencereyi kontrol et!")
        
        # Sınıf haritalaması
        class_names = {0: 'REAL (0)', 1: 'FAKE (1)'}
        
        # 4 resmi ekrana çizdirme döngüsü
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.suptitle("Data Loader Doğrulama Paneli (İlk 4 Görsel)", fontsize=14, fontweight='bold')
        
        for i in range(4):
            # PyTorch formatını (3, 224, 224) Matplotlib formatına (224, 224, 3) çevir
            img = images[i].numpy().transpose((1, 2, 0))
            
            # ImageNet normalizasyonunu tersine çevir (Renkler düzgün çıksın diye)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {class_names[labels[i].item()]}", fontsize=11, color='darkblue', fontweight='bold')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\n[X] Test Hatası: {e}")
        print("-> Lütfen veri klasörlerinin doğru yerde olduğunu kontrol et: ./data/train ve ./data/test")