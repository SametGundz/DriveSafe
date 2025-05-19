# Driver Drowsiness Detection (Sürücü Uykululuk Tespiti)

Bu proje, sürücülerin uykululuk durumunu gerçek zamanlı olarak tespit eden bir sistemdir. ETH-XGaze bakış tahmin modeli kullanılarak sürücünün göz hareketleri, baş pozisyonu ve bakış yönü analiz edilir. Modüler yazılım mimarisi ve performans optimizasyonları sayesinde yüksek verimlilikle çalışır.

## Özellikler

- **Gerçek zamanlı yüz tespiti ve izleme**: MediaPipe Face Mesh kullanarak yüksek hassasiyetli yüz işaretleri tespiti
- **ETH-XGaze entegrasyonu**: Gelişmiş bakış yönü tahmini için son teknoloji modelin kullanımı
- **Uykululuk tespiti**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio) ve PERCLOS (Percentage of Eye Closure) metrikleri ile gelişmiş analiz
- **Performans optimizasyonları**: Yapılandırılabilir kare atlama ve önbelleğe alma stratejileri
- **Modüler mimari**: Bakım ve genişletme kolaylığı sağlayan bileşen tabanlı yapı
- **Çoklu giriş desteği**: Web kamerası veya video dosyası girişi
- **Çoklu model desteği**: ONNX ve PyTorch model formatları
- **FPS görüntüleme**: Anlık performans değerlendirmesi için kare hızı gösterimi
- **Kapsamlı görselleştirme**: Bakış yönü, baş duruşu ve yüz işaretleri için zengin görsel gösterimler
- **Grafik analiz arayüzü**: Metrikler için zaman serisi grafikleri ve genişletilebilir istatistik pencereleri

## Mimari Genel Bakış

Sistem, aşağıdaki ana bileşenlerden oluşan modüler bir mimariye sahiptir:

### Veri İşleme Bileşenleri
- **FaceLandmarkDetector**: Yüz tespiti ve 468 landmark noktası çıkarımı
- **Facial Metrics**: EAR ve MAR gibi metrik hesaplamaları
- **HeadPoseEstimator**: Yüz landmarklarından baş pozisyonu tahmini
- **GazeDetector**: Bakış yönü tespiti ve görselleştirme
- **MediaPipeHelper**: Tüm bileşenleri birleştiren entegrasyon arayüzü

### Kullanıcı Arayüzü Bileşenleri
- **VideoPanel**: Kamera görüntüsü ve analiz görselleştirmeleri
- **MetricsPanel**: EAR, MAR ve PERCLOS metrikleri gösterimi
- **ChartPanel**: Zaman serisi grafikleri ve istatistikler
- **ControlPanel**: Uygulama kontrolü ve izleme seçenekleri
- **SettingsDialog**: Uygulama parametrelerinin yapılandırılması

## Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- Numpy
- ONNX (opsiyonel)
- ONNXRuntime (opsiyonel)
- MediaPipe
- PyQt6
- PyQtChart

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

#### 3. Model Varlığını Doğrulama

Model dosyasının düzgün bir şekilde yerleştirildiğini doğrulamak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
ls -la models/
```

Dosya listesinde `eth_xgaze_model.pth` veya `eth_xgaze_model.onnx` dosyasını görmelisiniz.

### Projeyi Kurma

1. Repoyu klonlayın:
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness.git
   cd driver-drowsiness
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Yapılandırma dosyasını kontrol edin:
   ```bash
   cat config/config.yaml
   ```
   
   Yapılandırma dosyasında bulunan `frame_skip` değeri, bakış tespiti sırasında kaç karede bir tahmin yapılacağını belirler. Bu değeri performans ihtiyaçlarınıza göre ayarlayabilirsiniz.

## Kullanım

### Uygulamayı Çalıştırma

Ana uygulamayı başlatmak için:

```bash
python run.py
```

Bu komut, sürücü uykululuk tespit sisteminin grafiksel kullanıcı arayüzünü başlatır.

### Bakış Tahmini Demo

Web kamerası veya video dosyasından gerçek zamanlı bakış tahmini için:

```bash
python examples/gaze_estimation_demo.py --device cpu --input webcam
# veya bir video dosyası için:
python examples/gaze_estimation_demo.py --device cpu --input path/to/video.mp4
# veya normalize edilmiş yüz görüntüsünü göstermek için:
python examples/gaze_estimation_demo.py --device cpu --input webcam --show_normalized
```

### Uykululuk Tespiti Demo

Sürücünün uykululuk durumunu tespit etmek için:

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

## Modüler MediaPipe Mimarisi

Bu projede, performansı ve kod organizasyonunu iyileştirmek için MediaPipe işlevleri modüler bir yapıda yeniden düzenlenmiştir. Bu değişiklikler, kodun okunabilirliğini, bakımını ve genişletilebilirliğini önemli ölçüde artırmaktadır.

### Modüller ve Sorumlulukları

- **FaceLandmarkDetector**: 
  - Yüz tespiti ve izleme
  - 468 facial landmark noktasının çıkarımı
  - Özel nokta gruplarına (göz, ağız) erişim
  - Landmark görselleştirme işlevleri

- **Facial Metrics**: 
  - EAR (Göz Açıklık Oranı) hesaplama
  - MAR (Ağız Açıklık Oranı) hesaplama
  - Mesafe hesaplama işlevleri
  - Nokta tabanlı ölçüm algoritmaları

- **HeadPoseEstimator**: 
  - Facial landmark'lardan kafa pozisyonu hesaplama
  - Pitch, yaw ve roll açılarını tespit etme
  - 3D-2D nokta haritalama işlevleri
  - Baş pozisyonu görselleştirme (eksenler veya küp)

- **GazeDetector**: 
  - ETH-XGaze modeli kullanarak bakış yönü tahmini
  - Yüz normalizasyonu ve görüntü ön işleme
  - Bakış vektörü görselleştirme
  - Kare atlama optimizasyonu ile performans iyileştirme

- **MediaPipeHelper**: 
  - Tüm modüler bileşenleri birleştiren entegrasyon katmanı
  - Eski arayüz ile uyumlu birleşik API
  - Bileşen oluşturma ve yönetme
  - Kaynak yönetimi işlevleri

### Performans İyileştirmeleri

- **Kare Atlama**: Bakış tespiti, `frame_skip` parametresi ile kontrol edilen belirli aralıklarla çalışır
- **Yapılandırılabilir Performans**: `config.yaml` dosyasında `frame_skip` değeri kolayca ayarlanabilir
- **Seçici İşleme**: Her karede değil, yalnızca belirli aralıklarla ağır işlemler gerçekleştirilir
- **Önbelleğe Alma**: Tahminler arasında son tahmin edilen bakış vektörü yeniden kullanılır
- **FPS Ölçümü**: Anlık FPS gösterimi ile performans değerlendirmesi yapılabilir

### Kullanım Örnekleri

#### Yeni Modüler API Kullanımı (Önerilen)

```python
# Birleşik arayüz kullanımı
from src.utils import MediaPipeHelper, get_mediapipe_helper

# Helper örneği alın
mp_helper = get_mediapipe_helper()

# Yüz landmarkları tespit edin
landmarks, face_detected = mp_helper.detect_face_landmarks(frame)

# Göz ve ağız metrikleri hesaplayın
left_eye = mp_helper.get_eye_landmarks(landmarks, left_eye=True)
right_eye = mp_helper.get_eye_landmarks(landmarks, left_eye=False)
left_ear = mp_helper.get_eye_aspect_ratio(left_eye)
right_ear = mp_helper.get_eye_aspect_ratio(right_eye)
ear = (left_ear + right_ear) / 2.0
mar = mp_helper.get_mouth_aspect_ratio(landmarks)

# Baş duruşu ve bakış yönünü görselleştirin
frame = mp_helper.visualize_head_pose(frame, landmarks, visualization_type='cube')
frame, normalized_face = mp_helper.visualize_gaze(
    frame, landmarks, ear_value=ear, ear_threshold=0.21, frame_skip=3
)
```

#### Ayrı Modülleri Kullanma

```python
# Bireysel bileşenleri doğrudan kullanma
from src.utils.face_landmark_detector import FaceLandmarkDetector
from src.utils.facial_metrics import get_eye_aspect_ratio, get_mouth_aspect_ratio
from src.utils.head_pose_estimator import HeadPoseEstimator
from src.utils.gaze_detector import GazeDetector

# Bileşenleri ayrı ayrı oluşturma
face_detector = FaceLandmarkDetector()
head_pose_estimator = HeadPoseEstimator()
gaze_detector = GazeDetector()

# Yüz landmarkları tespit etme
landmarks, face_detected = face_detector.detect_face_landmarks(frame)

# Göz landmarkları ve metrikleri
left_eye = face_detector.get_eye_landmarks(landmarks, left_eye=True)
right_eye = face_detector.get_eye_landmarks(landmarks, left_eye=False)
left_ear = get_eye_aspect_ratio(left_eye)
right_ear = get_eye_aspect_ratio(right_eye)

# Baş duruşu ve bakış yönü işlemleri
rotation_matrix, angles = head_pose_estimator.calculate_head_pose(landmarks, frame)
frame = head_pose_estimator.visualize_head_pose(frame, landmarks)
frame, normalized_face = gaze_detector.visualize_gaze(frame, landmarks, frame_skip=3)
```

## Yapılandırma ve Özelleştirme

Sistem, `config/config.yaml` ve `config/ui_config.yaml` dosyaları aracılığıyla yapılandırılabilir. Bu dosyalarda aşağıdaki parametreleri ayarlayabilirsiniz:

### Kamera Ayarları
- Kamera cihazı ID'si
- Çözünürlük (genişlik, yükseklik)
- FPS değeri

### Algılama Parametreleri
- Göz kapalılık eşiği (EAR threshold)
- Ağız açıklık eşiği (MAR threshold)
- PERCLOS hesaplama penceresi
- Bakış tespiti için kare atlama (frame_skip)

### Kullanıcı Arayüzü Ayarları
- Pencere boyutları
- Font boyutları ve tipleri
- Grafik aralıkları ve süreleri

Örnek yapılandırma değişikliği:

```yaml
# config/config.yaml
detection:
  ear_threshold: 0.21  # Göz kapalılık eşiği
  mar_threshold: 0.70  # Ağız açıklık eşiği
  perclos_window_sec: 60  # PERCLOS hesaplama penceresi (saniye)
  gaze:
    frame_skip: 5  # Bakış tespiti kare atlama değeri
```

## Test ve Değerlendirme

### Ünite Testleri

Ünite testlerini çalıştırmak için:

```bash
python tests/run_tests.py
# veya belirli bir test dosyası için:
python -m unittest tests/utils/test_face_landmark_detector.py
```

### Performans Değerlendirmesi

Uygulama çalışırken FPS (Frames Per Second) değeri ekranda görüntülenir. Bu değer, sistemin gerçek zamanlı performansını gösterir. Düşük FPS değerleri gözlemlerseniz, aşağıdaki optimizasyonları deneyebilirsiniz:

1. `config.yaml` dosyasında `frame_skip` değerini artırın
2. Daha düşük kamera çözünürlüğü kullanın
3. Daha hafif model kullanın (ONNX modeli genellikle daha hızlıdır)

## Sorun Giderme

### Gaze Tahmini Çalışmıyor

Eğer "cannot reshape array of size X into shape (6,1,2)" benzeri bir hata alıyorsanız:
- MediaPipe'dan gelen landmark'ların doğru şekilde işlenmediği anlamına gelir
- `detect_face_landmarks` fonksiyonunun düzgün çalıştığından emin olun
- Yüzün kamera görüş alanında tam olarak görünür olduğunu kontrol edin

### Model Dosyası Bulunamadı

Eğer "ETH-XGaze model dosyası bulunamadı" hatası alıyorsanız, şu konumlardan birinde model dosyasının bulunduğundan emin olun:
- models/eth_xgaze_model.pth
- models/eth_xgaze_model.pth.tar
- models/pretrained/eth_xgaze.pth
- models/pretrained/eth_xgaze.pth.tar

### Performans Sorunları

Gaze tespiti sırasında performans sorunları yaşıyorsanız:
- `config.yaml` dosyasındaki `frame_skip` değerini artırın (3-5 arası iyi bir başlangıç noktasıdır)
- Kamera çözünürlüğünü düşürün
- İhtiyaç duymadığınız görselleştirme özelliklerini kapatın (Baş duruşu, Yüz işaretleri vb.)
- Ağır işlemler için GPU kullanımını etkinleştirmeyi deneyin

## Katkıda Bulunma

Katkıda bulunmak için, lütfen aşağıdaki adımları izleyin:

1. Repoyu fork edin
2. Feature branch oluşturun: `git checkout -b feature/amazing-feature`
3. Değişikliklerinizi kaydedin: `git commit -m 'Add amazing feature'`
4. Branch'inize push yapın: `git push origin feature/amazing-feature`
5. Pull Request oluşturun

## MediaPipe Landmark İndeksleri

Gaze tahmini için kullanılan özel landmark indeksleri şunlardır:

- **Sol göz köşeleri**: 33, 133
- **Sağ göz köşeleri**: 362, 263
- **Burun**: 4, 5
- **Çene**: 199

Bu landmark'lar, MediaPipe Face Mesh tarafından sağlanan 468 noktanın özel bir alt kümesidir. Doğru gaze tahmini için bu indekslerin doğru şekilde çıkarılması gerekmektedir.

MediaPipe Face Mesh landmark indeksleriyle ilgili tam belgelendirme için [Google MediaPipe Face Mesh sayfasına](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) bakabilirsiniz.

## ETH-XGaze Entegrasyonu

Bu proje, ETH-XGaze bakış tahmini modelini kullanmaktadır. ETH-XGaze'in orijinal çalışmasıyla uyumlu olması için aşağıdaki bileşenler uyarlanmıştır:

1. **Yüz Normalizasyonu**: Yüz görselleri, ETH-XGaze projesindeki `normalizeData_face` fonksiyonu baz alınarak normalize edilir.
2. **Model Yapısı**: GazeResNet, ETH-XGaze projesindeki ResNet50 mimarisi temel alınarak oluşturulmuştur.
3. **Model Yükleme**: ModelLoader, ETH-XGaze modelleriyle uyumlu çalışacak şekilde tasarlanmıştır.
4. **6-Nokta Landmark İşleme**: ETH-XGaze'in 6 noktalı yüz modeli ile uyumlu çalışması için özel landmark işleme rutinleri kullanılmıştır.

## Lisans

Bu proje, açık kaynak [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## Referanslar

- [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation](https://ait.ethz.ch/projects/2020/ETH-XGaze/)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)