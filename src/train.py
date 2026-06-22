import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_data_loaders
from src.model import DeepfakeDetector
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train_and_save_model():
    print("=" * 70)
    print(" DERİN ÖĞRENİM EĞİTİM MOTORU - FULL DATASET WITH GRADIENT FLOW")
    print("=" * 70)
    
    # Hiperparametreler
    BATCH_SIZE = 32
    EPOCHS = 10  # 50'den 10'e düşürüldü (hızlı test için yeterli)
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/detector_model.pth"
    
    # ====================================================================
    # GPU/CPU DURUM KONTROL VE KONFIRMASYONU
    # ====================================================================
    print(f"\n [GPU CHECK]")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Cihazı: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"   GPU KULLANILACAK")
    else:
        print(f"    GPU BULUNMADI - CPU'DA ÇALIŞACAK")
    print(f"  Eğitim Cihazı: {str(DEVICE).upper()}")
    print(f"  Model Kayıt Yolu: {MODEL_PATH}\n")
    
    # Veri yükle (tam veri seti)
    print("[*] Tam veri seti (train + test) yükleniyor...")
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE, num_workers=0)
    
    # Modeli oluştur ve eğitim moduna al
    model = DeepfakeDetector().to(DEVICE)
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_test_acc = 0.0
    best_epoch = 0
    
    print(f"[>] {EPOCHS} Epoch boyunca eğitim başlatılıyor...\n")
    
    for epoch in range(EPOCHS):
        # ====================================================================
        # EĞİTİM AŞAMASI (TRAINING PHASE) - Gradyan Akışı Aktif
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAİN]", leave=True) as train_pbar:
            for images, labels in train_pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                labels = 1 - labels
                
                # İleri Besleme
                optimizer.zero_grad()
                outputs = model(images)  # Shape: [batch_size, 2] (REAL vs FAKE)
                loss = criterion(outputs, labels)
                
                # Geriye Yayılım (Gradient Flow - Saliency Map için kritik!)
                loss.backward()
                optimizer.step()
                
                # Metrikler
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy().tolist())
                train_labels.extend(labels.cpu().numpy().tolist())
                
                train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds) * 100
        train_f1 = f1_score(train_labels, train_preds, average='weighted') * 100
        
        # ====================================================================
        # TEST AŞAMASI (VALIDATION PHASE)
        # ====================================================================
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            with tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TEST]", leave=True) as test_pbar:
                for images, labels in test_pbar:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    labels = 1 - labels
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    test_preds.extend(predicted.cpu().numpy().tolist())
                    test_labels.extend(labels.cpu().numpy().tolist())
                    
                    test_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(test_labels, test_preds) * 100
        test_f1 = f1_score(test_labels, test_preds, average='weighted') * 100
        
        # Epoch Sonuçları
        print(f"\n[Epoch {epoch+1:02d}/{EPOCHS}]")
        print(f"  TRAİN → Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | F1-Score: {train_f1:.2f}%")
        print(f"  TEST  → Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}% | F1-Score: {test_f1:.2f}%")
        
        # Best Model Kayıt (Test Accuracy'sine göre)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Model kaydedildi! (En İyi Test Doğruluğu: {best_test_acc:.2f}%)")
    
    # Final Özet
    print("\n" + "=" * 70)
    print(" EĞİTİM TAMAMLANDI!")
    print(f"  En İyi Test Doğruluğu: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Model Dosya Yolu: {MODEL_PATH}")
    print("  → app.py tarafından yüklenmeye hazır")
    print("=" * 70)

if __name__ == '__main__':
    train_and_save_model()