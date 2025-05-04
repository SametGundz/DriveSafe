# Sürücü Uykululuk Tespit Sistemi (Driver Drowsiness Detection System)

Bu proje, sürücü uykululuğunu gerçek zamanlı olarak tespit eden bir sistem sunar. Kamera üzerinden alınan görüntülerde sürücünün yüz ve göz hareketlerini izleyerek uykululuk belirtilerini tespit eder ve uyarılar oluşturur.

## Özellikler

- **Göz Kapalılık Oranı (EAR)**: Gözlerin kapalı olup olmadığını tespit eder
- **Ağız Açıklık Oranı (MAR)**: Esneme tespiti için ağız açıklığını izler
- **Baş Pozisyonu Tahmini**: Sürücünün baş pozisyonunu takip eder
- **PERCLOS (Percentage of Eye Closure)**: Belirli bir zaman diliminde gözlerin kapalı olduğu süre yüzdesini hesaplar
- **Bakış Yönü Tespiti**: Sürücünün nereye baktığını tespit eder
- **Karar Mekanizması**: Tüm bu verileri birleştirerek uykululuk seviyesini belirler
- **Görselleştirme**: Tespit sonuçlarını gerçek zamanlı olarak gösterir
- **Video Kayıt**: İsteğe bağlı olarak video kaydı alabilir

## Çalışma Prensibi

### Yüz İşaretleri Tespiti
Sistem, MediaPipe Face Mesh modeli kullanarak yüzdeki 468 adet işaret noktasını (landmarks) tespit eder. Bu işaret noktaları, göz, ağız, burun, yüz konturu gibi yüzün önemli bölgelerini temsil eder.

### Göz Kapalılık Oranı (EAR - Eye Aspect Ratio)
EAR, göz işaret noktaları kullanılarak hesaplanan bir orandır ve gözün açıklık derecesini ölçer:

```
EAR = (V1 + V2 + V3) / (3 * H)
```

Burada:
- V1, V2, V3: Gözdeki üç farklı dikey mesafe (üst ve alt göz kapağı arasındaki mesafeler)
- H: Göz köşeleri arasındaki yatay mesafe

EAR değeri, göz açıkken daha yüksek, kapalıyken daha düşüktür. Belirli bir eşik değerinin altına düştüğünde (tipik olarak 0.21), gözün kapalı olduğu tespit edilir.

### Ağız Açıklık Oranı (MAR - Mouth Aspect Ratio)
MAR, ağız işaret noktaları kullanılarak hesaplanan bir orandır ve ağzın açıklık derecesini ölçer. İç dudak noktaları kullanılarak hesaplanır:

```
MAR = (V_ortalaması) / H
```

Burada:
- V_ortalaması: Ağzın üst ve alt dudakları arasındaki dikey mesafelerin ortalaması
- H: Ağzın köşeleri arasındaki yatay mesafe

MAR değeri, ağız açıkken (esneme durumunda) yükselir. Belirli bir eşik değerinin üzerine çıktığında (tipik olarak 0.65), esneme tespit edilir.

### PERCLOS (Percentage of Eye Closure)
PERCLOS, belirli bir zaman diliminde (tipik olarak 1-3 dakika) gözlerin kapalı olduğu sürenin yüzdesidir:

```
PERCLOS = (Gözün kapalı olduğu kareler / Toplam kare sayısı) * 100
```

Yüksek PERCLOS değeri (tipik olarak %15'in üzeri), sürücü yorgunluğunun önemli bir göstergesidir.

## Kurulum

1. Gerekli bağımlılıkları yükleyin:
   ```bash
   pip install opencv-python numpy mediapipe pyyaml
   ```

2. Projeyi klonlayın veya indirin:
   ```bash
   git clone https://github.com/kullaniciadi/driver-drowsiness.git
   cd driver-drowsiness
   ```

3. 3D yüz modelini oluşturun:
   ```bash
   python create_3d_model.py
   ```

## Kullanım

Sistemi başlatmak için aşağıdaki seçeneklerden birini kullanabilirsiniz:

### 1. GUI ile Kullanım

Grafiksel arayüz ile sistemi başlatmak için:

```bash
# Ana dizinden:
./run_gui.py

# Veya Python ile:
python run_gui.py

# Alternatif olarak:
python src/main.py
```

GUI arayüzü şu özelliklere sahiptir:
- Kamera görüntüsünün canlı görüntülenmesi
- EAR, MAR ve PERCLOS değerlerinin gerçek zamanlı göstergeleri
- Baş pozisyonu ve bakış yönü bilgileri
- Gerçek zamanlı veri grafiği
- Ayarlar menüsü ile parametrelerin düzenlenmesi
- Menü ve araç çubuğu üzerinden tüm işlevlere erişim

### 2. Komut Satırı ile Kullanım

Komut satırı arayüzü ile sistemi başlatmak için:

```bash
python run.py [SEÇENEKLER]
```

### Komut Satırı Seçenekleri

- `--camera KAMERA_INDEKSI`: Kullanılacak kamera indeksi (varsayılan: 0)
- `--width GENİŞLİK`: Kamera çerçeve genişliği (varsayılan: 640)
- `--height YÜKSEKLİK`: Kamera çerçeve yüksekliği (varsayılan: 480)
- `--fps FPS`: Kamera kare hızı (varsayılan: 30)
- `--show-fps`: Ekranda FPS göstergesini görüntüler
- `--record`: Video kaydını etkinleştirir
- `--log-level LEVEL`: Günlük seviyesi (debug, info, warning, error) (varsayılan: info)
- `--config DOSYA_YOLU`: Konfigürasyon dosyası yolu (varsayılan: config/config.yaml)

### Örnek Kullanım

```bash
python run.py --camera 0 --width 800 --height 600 --show-fps --record
```

### Klavye Kısayolları

Uygulama çalışırken aşağıdaki klavye kısayollarını kullanabilirsiniz:

- `q`: Uygulamadan çıkış
- `r`: Video kaydını başlat/durdur
- `d`: Algılamayı etkinleştir/devre dışı bırak
- `h`: Yardım menüsünü göster/gizle
- `s`: Ekran görüntüsü al

## Teknik Detaylar

### Yüz ve Göz Tespiti
MediaPipe Face Mesh modeli, toplamda 468 yüz işaret noktası (landmarks) kullanır. Bu sistemde:

- **Göz İşaret Noktaları**:
  - Sol göz: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
  - Sağ göz: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

- **Ağız İşaret Noktaları**:
  - Dış dudak: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
  - İç dudak: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

### EAR Hesaplama Algoritması
Göz Açıklık Oranı (EAR), gözdeki özel nokta çiftleri kullanılarak hesaplanır. Bu hesaplamada:

1. Her bir göz için üç farklı dikey mesafe ölçülür:
   - Göz üst kenarı ve alt kenarı arasındaki dikey mesafeler
2. Göz köşeleri arasındaki yatay mesafe ölçülür
3. EAR = Dikey mesafelerin ortalaması / Yatay mesafe

### MAR Hesaplama Algoritması
Ağız Açıklık Oranı (MAR), iç dudak işaret noktaları kullanılarak hesaplanır. Bu hesaplamada:

1. İç dudak köşeleri arasındaki yatay mesafe ölçülür (H)
2. İç dudağın üst ve alt noktaları arasında çeşitli dikey mesafeler ölçülür:
   - Orta dikey mesafe (dudak ortasında)
   - Sol taraf dikey mesafe
   - Sağ taraf dikey mesafe
3. Dikey mesafelerin ortalaması hesaplanır (V_ort)
4. MAR = V_ort / H

Bu yaklaşım, ağzın farklı bölgelerindeki açıklıkları dikkate alarak daha doğru bir MAR değeri elde etmeyi sağlar.

### Karar Verme Mekanizması
Uykululuk tespiti, aşağıdaki parametrelerin kombinasyonu kullanılarak yapılır:

1. **EAR Eşik Değeri**: Tipik olarak 0.21-0.25 arası, bu değerin altında gözler kapalı kabul edilir
2. **MAR Eşik Değeri**: Tipik olarak 0.6-0.7 arası, bu değerin üstünde esneme tespit edilir
3. **PERCLOS Eşik Değeri**: Tipik olarak %15, bu değerin üstünde sürücü yorgun kabul edilir
4. **Göz Kapanma Süresi**: Gözlerin kapalı kaldığı süre, mikro uyku tespiti için kullanılır

## Proje Yapısı

```
driver-drowsiness/
├── config/
│   └── config.yaml             # Konfigürasyon dosyası
├── models/
│   └── headpose_3d_model.npy   # 3D yüz modeli
├── recordings/                 # Kaydedilen videolar
├── screenshots/                # Ekran görüntüleri
├── logs/                       # Log dosyaları
├── src/
│   ├── detection/              # Algılama modülleri
│   ├── decision/               # Karar modülleri
│   ├── ui/                     # Grafiksel arayüz modülleri
│   │   ├── widgets.py          # Özel arayüz bileşenleri
│   │   └── main_window.py      # Ana pencere uygulaması
│   ├── utils/                  # Yardımcı modüller
│   └── main.py                 # GUI ana giriş noktası
├── create_3d_model.py          # 3D model oluşturucu
├── run.py                      # Komut satırı başlatıcı betik
└── README.md                   # Bu dosya
```

## Konfigürasyon

Sistem davranışını özelleştirmek için `config/config.yaml` dosyasını düzenleyebilirsiniz. Bu dosyada şunları ayarlayabilirsiniz:

- Kamera parametreleri
- Algılama eşikleri
- Uykululuk değerlendirme parametreleri
- Görselleştirme seçenekleri
- Günlükleme ayarları

## Lisans

Bu proje [MIT lisansı](LICENSE) altında lisanslanmıştır.

## İletişim

Sorularınız veya geri bildirimleriniz için iletişime geçin: email@example.com

## Gereksinimler

- Python 3.8 veya üzeri
- Bağımlılıklar:
  ```bash
  pip install -r requirements.txt
  ```

Temel kütüphaneler:
- OpenCV
- NumPy
- MediaPipe
- PyYAML
- PyQt6
- PyQtChart
