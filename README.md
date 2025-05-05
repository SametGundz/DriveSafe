# Driver Drowsiness Detection (Sürücü Uykusu Tespiti)

Bu proje, sürücülerin uykululuk durumunu gerçek zamanlı olarak tespit eden bir sistemdir. ETH-XGaze bakış tahmin modeli kullanılarak sürücünün göz hareketleri, baş pozisyonu ve bakış yönü analiz edilir.

## Özellikler

- Gerçek zamanlı yüz tespiti ve izleme
- ETH-XGaze modelini kullanarak bakış yönü tahmini
- Göz kapalılık durumu ve uykululuk tespiti
- Web kamerası veya video dosyası girişi desteği
- ONNX ve PyTorch model formatları desteği

## Kurulum

### ETH-XGaze Modelini İndirme ve Kurulum

ETH-XGaze modeli, bu projenin çalışması için gerekli olan temel bir bileşendir. Model dosyaları büyük boyutta olduğu için genellikle Git repositorylerine dahil edilmez ve .gitignore'a eklenir. Aşağıdaki adımları izleyerek ETH-XGaze modelini projenize ekleyebilirsiniz:

#### 1. ETH-XGaze Modelini İndirme

ETH-XGaze modelini şu kaynaklardan edinebilirsiniz:

##### A. Resmi ETH-XGaze Deposundan İndirme:

1. [ETH-XGaze resmi GitHub deposunu](https://github.com/xucong-zhang/ETH-XGaze) klonlayın:
   ```bash
   git clone https://github.com/xucong-zhang/ETH-XGaze.git
   ```

2. Eğitilmiş modeli [ETH-XGaze proje sayfasından](https://ait.ethz.ch/projects/2020/ETH-XGaze/) indirin. Modele erişmek için bir form doldurmanız gerekebilir.

3. İndirilen model dosyasını (`epoch_24_ckpt.pth.tar` veya benzer isimde bir dosya) `ETH-XGaze/ckpt/` klasörüne yerleştirin.

##### B. Önceden Dönüştürülmüş Modeli İndirme (Alternatif):

Eğer dönüştürülmüş bir model dosyası kullanmak isterseniz, şu kaynaklardan edinebilirsiniz:
- [Google Drive](https://drive.google.com/drive/folders/MODEL_ID) veya benzer bir kaynaktan önişlenmiş ETH-XGaze modelini indirebilirsiniz.

#### 2. Model Dosyasını Projeye Ekleme

1. `models` klasörü oluşturun (zaten varsa bu adımı atlayın):
   ```bash
   mkdir -p models
   ```

2. İndirdiğiniz model dosyasını `models` klasörüne kopyalayın veya taşıyın:

   **A) Orijinal ETH-XGaze modeli için:**
   ```bash
   # Orijinal model dosyasını dönüştürüp models klasörüne kaydedin
   python scripts/convert_ethxgaze_model.py ETH-XGaze/ckpt/epoch_24_ckpt.pth.tar models/eth_xgaze_model.pth
   # ONNX formatında da dönüştürebilirsiniz (opsiyonel)
   python scripts/convert_ethxgaze_model.py ETH-XGaze/ckpt/epoch_24_ckpt.pth.tar models/eth_xgaze_model.onnx --export_onnx
   ```

   **B) Önceden dönüştürülmüş model için:**
   ```bash
   # İndirdiğiniz dosyayı models klasörüne kopyalayın
   cp /indirme/konumu/eth_xgaze_model.pth models/
   # veya
   cp /indirme/konumu/eth_xgaze_model.onnx models/
   ```

#### 4. Model Varlığını Doğrulama

Model dosyasının düzgün bir şekilde yerleştirildiğini doğrulamak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
ls -la models/
```

Dosya listesinde `eth_xgaze_model.pth` veya `eth_xgaze_model.onnx` dosyasını görmelisiniz.

#### 5. Test Etme

Modelin düzgün çalıştığını test etmek için, örnek uygulamayı çalıştırın:

```bash
python examples/gaze_estimation_demo.py --device cpu --input webcam
```

Eğer model doğru yüklendiyse, uygulama sorunsuz çalışacaktır.

### Gereksinimler

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- Numpy
- ONNX (opsiyonel, ONNX model kullanımı için)
- ONNXRuntime (opsiyonel, ONNX model kullanımı için)
- MediaPipe (yüz tespiti ve landmark tespiti için)

### Kurulum Adımları

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness.git
   cd driver-drowsiness
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. ETH-XGaze modellerini indirin veya dönüştürün:
   ```bash
   # Eğer orijinal ETH-XGaze reposu varsa:
   python scripts/convert_ethxgaze_model.py ../ETH-XGaze/ckpt/epoch_24_ckpt.pth.tar models/eth_xgaze_model.pth
   # Ayrıca ONNX modeli oluşturmak için:
   python scripts/convert_ethxgaze_model.py ../ETH-XGaze/ckpt/epoch_24_ckpt.pth.tar models/eth_xgaze_model.onnx --export_onnx
   ```

## Kullanım

### Bakış Tahmini Demo

Bu uygulama, web kamerası veya video dosyasından gerçek zamanlı bakış tahmini yapar:

```bash
python examples/gaze_estimation_demo.py --device cpu --input webcam
# veya bir video dosyası için:
python examples/gaze_estimation_demo.py --device cpu --input path/to/video.mp4
# veya normalize edilmiş yüz görüntüsünü göstermek için:
python examples/gaze_estimation_demo.py --device cpu --input webcam --show_normalized
```

### Uykululuk Tespiti Demo

Bu uygulama, sürücünün uykululuk durumunu tespit eder:

```bash
python examples/drowsiness_detection.py --device cpu --input webcam
# veya bir video dosyası için:
python examples/drowsiness_detection.py --device cpu --input path/to/video.mp4
```

### Model Dönüştürücü

ETH-XGaze model formatlarını dönüştürmek için:

```bash
python scripts/convert_ethxgaze_model.py <kaynak_model> <hedef_model> [--export_onnx] [--device cpu|cuda]
```

### MediaPipe Landmark İndeksleri

Gaze tahmini için kullanılan özel landmark indeksleri şunlardır:

- **Sol göz köşeleri**: 33, 133
- **Sağ göz köşeleri**: 362, 263
- **Burun**: 4, 5

Bu landmark'lar, MediaPipe Face Mesh tarafından sağlanan 468 noktanın özel bir alt kümesidir. Doğru gaze tahmini için bu indekslerin doğru şekilde çıkarılması gerekmektedir.

MediaPipe Face Mesh landmark indeksleriyle ilgili tam belgelendirme için [Google MediaPipe Face Mesh sayfasına](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) bakabilirsiniz.

## ETH-XGaze Entegrasyonu

Bu proje, ETH-XGaze bakış tahmini modelini kullanmaktadır. ETH-XGaze'in orijinal çalışmasıyla uyumlu olması için aşağıdaki bileşenler uyarlanmıştır:

1. **Yüz Normalizasyonu**: Yüz görselleri, ETH-XGaze projesindeki `normalizeData_face` fonksiyonu baz alınarak normalize edilir.
2. **Model Yapısı**: GazeResNet, ETH-XGaze projesindeki ResNet50 mimarisi temel alınarak oluşturulmuştur.
3. **Model Yükleme**: ModelLoader, ETH-XGaze modelleriyle uyumlu çalışacak şekilde tasarlanmıştır.

## Test

Ünite testlerini çalıştırmak için:

```bash
python -m unittest discover tests
# veya belirli bir test dosyası için:
python -m unittest tests.test_gaze_estimator
```

## Sorun Giderme

### Gaze Tahmini Çalışmıyor

Eğer "cannot reshape array of size X into shape (6,1,2)" benzeri bir hata alıyorsanız, MediaPipe'dan gelen landmark'ların doğru şekilde işlenmediği anlamına gelir. process_landmarks fonksiyonunun düzgün çalıştığından ve doğru indekslerin kullanıldığından emin olun.

### Model Dosyası Bulunamadı

Eğer "ETH-XGaze model dosyası bulunamadı" hatası alıyorsanız, şu konumlardan birinde model dosyasının bulunduğundan emin olun:
- models/eth_xgaze_model.pth
- models/eth_xgaze_model.pth.tar
- models/pretrained/eth_xgaze.pth
- models/pretrained/eth_xgaze.pth.tar

## Lisans

Bu proje, açık kaynak [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## Referanslar

- [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation](https://ait.ethz.ch/projects/2020/ETH-XGaze/)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)