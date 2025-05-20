# Driver Drowsiness Detection (Sürücü Uykululuk Tespiti)

Bu proje, sürücülerin uykululuk durumunu gerçek zamanlı olarak tespit eden bir sistemdir. ETH-XGaze bakış tahmin modeli kullanılarak sürücünün göz hareketleri, baş pozisyonu ve bakış yönü analiz edilir. Modüler yazılım mimarisi ve performans optimizasyonları sayesinde yüksek verimlilikle çalışır.

## Özellikler

- **Gerçek zamanlı yüz tespiti ve izleme**: MediaPipe Face Mesh kullanarak yüksek hassasiyetli yüz işaretleri tespiti
- **ETH-XGaze entegrasyonu**: Gelişmiş bakış yönü tahmini için son teknoloji modelin kullanımı
- **Uykululuk tespiti**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio) ve PERCLOS (Percentage of Eye Closure) metrikleri ile gelişmiş analiz
- **Bakış bölgesi (Gaze Zone) analizi**: Sürücünün araç içinde nereye baktığını (dikiz aynası, direksiyon, yan pencereler vb.) tespit eden ve raporlayan sistem
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
- **GazeZoneDetector**: Bakış yönü açılarından araç içi bölge tespiti ve analizi
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

### Bakış Bölgesi Tespiti Demo

Sürücünün araç içinde nereye baktığını (dikiz aynası, direksiyon, yan pencere vb.) tespit etmek için:

```bash
python tests/test_gaze_zone_detector.py --camera 0
# veya farklı parametreler ile:
python tests/test_gaze_zone_detector.py --camera 0 --history 15 --stability 0.7 --show_landmarks
```

Bu demo, sürücünün araç içinde 9 farklı bölgeye (sol yan pencere, direksiyon, dikiz aynası vb.) ne kadar süreyle baktığını tespit eder ve raporlar. Sonuçlar gerçek zamanlı olarak görüntülenir ve her bölge için toplam bakış süreleri hesaplanır.

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
  - PERCLOS (göz kapalılık yüzdesi) hesaplama
  - Göz kırpma ve esneme tespiti
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

- **GazeZoneDetector**:
  - Bakış açılarından araç içi bölge tespiti (9 farklı bölge)
  - Bakılan bölgelerin süre takibi ve istatistikleri 
  - Stabilize edilmiş bölge tespiti ile daha doğru sonuçlar
  - Modüler ve genişletilebilir bölge tanımlama yapısı

- **MediaPipeHelper**: 
  - Tüm modüler bileşenleri birleştiren entegrasyon katmanı
  - Eski arayüz ile uyumlu birleşik API
  - Bileşen oluşturma ve yönetme
  - Kaynak yönetimi işlevleri

- **DrowsinessDetector**:
  - Uykululuk durumunu tespit etme
  - PERCLOS, EAR, baş duruşu ve bakış metriklerini kullanarak analiz
  - Uyarı seviyesi belirleme
  - Sonuçların görselleştirilmesi

### Performans İyileştirmeleri

- **Kare Atlama**: Bakış tespiti, `frame_skip` parametresi ile kontrol edilen belirli aralıklarla çalışır
- **Yapılandırılabilir Performans**: `config.yaml` dosyasında `frame_skip` değeri kolayca ayarlanabilir
- **Seçici İşleme**: Her karede değil, yalnızca belirli aralıklarla ağır işlemler gerçekleştirilir
- **Önbelleğe Alma**: Tahminler arasında son tahmin edilen bakış vektörü yeniden kullanılır
- **FPS Ölçümü**: Anlık FPS gösterimi ile performans değerlendirmesi yapılabilir
- **Tip Belirteçleri**: Tüm modüllerde tutarlı tip belirteçleri (type hints) kullanılarak kod kalitesi artırılmıştır

## Modüler Mimari Kullanım Örnekleri

Projede bulunan modüler mimariyi farklı senaryolarda nasıl kullanabileceğinize dair kapsamlı örnekler aşağıda verilmiştir:

### 1. Temel Modüler API Kullanımı

Modüler API'yi kullanarak basit bir yüz tespiti ve metrik hesaplama işlemi gerçekleştirme:

```python
from src.utils import get_face_landmark_detector
from src.utils.facial_metrics import get_eye_aspect_ratio, get_mouth_aspect_ratio

# Görüntüyü al (OpenCV ile)
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# FaceLandmarkDetector örneği oluştur
detector = get_face_landmark_detector()

# Yüz landmarkları tespit et
landmarks, face_detected = detector.detect_face_landmarks(frame)

if face_detected:
    # Göz landmarkları al
    left_eye = detector.get_eye_landmarks(landmarks, left_eye=True)
    right_eye = detector.get_eye_landmarks(landmarks, left_eye=False)
    
    # EAR hesapla
    left_ear = get_eye_aspect_ratio(left_eye)
    right_ear = get_eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # MAR hesapla
    mouth = detector.get_mouth_landmarks(landmarks)
    mar = get_mouth_aspect_ratio(mouth)
    
    print(f"EAR: {avg_ear:.3f}, MAR: {mar:.3f}")
    
    # Landmarkları görselleştir
    viz_frame = detector.draw_facial_landmarks(
        frame.copy(), 
        left_eye + right_eye,  # Tüm göz landmarkları
        landmark_color=(0, 255, 0),
        landmark_radius=2
    )
    
    cv2.imshow("Facial Landmarks", viz_frame)
    cv2.waitKey(0)

# Kaynakları serbest bırak
detector.release()
cap.release()
cv2.destroyAllWindows()
```

### 2. MediaPipeHelper Tümleşik Arayüzünü Kullanma

Tüm bileşenleri tek bir arayüz üzerinden kullanarak uykululuk tespiti yapma:

```python
from src.utils import get_mediapipe_helper
from src.detection.drowsiness_detector import DrowsinessDetector
import cv2
import time

# MediaPipeHelper örneği oluştur
mp_helper = get_mediapipe_helper()

# Drowsiness detector oluştur
drowsiness_detector = DrowsinessDetector()

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # İşleme zamanını ölç
    start_time = time.time()
    
    # Yüz landmarkları tespit et
    landmarks, face_detected = mp_helper.detect_face_landmarks(frame)
    
    if face_detected:
        # Göz ve ağız metrikleri hesapla
        left_eye = mp_helper.get_eye_landmarks(landmarks, left_eye=True)
        right_eye = mp_helper.get_eye_landmarks(landmarks, left_eye=False)
        left_ear = mp_helper.get_eye_aspect_ratio(left_eye)
        right_ear = mp_helper.get_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = mp_helper.get_mouth_aspect_ratio(landmarks)
        
        # Baş duruşu hesapla
        roll, pitch, yaw = mp_helper.get_head_pose(landmarks, frame)
        
        # Tüm bilgileri drowsiness detector'a aktar
        drowsiness_result = drowsiness_detector.update(
            ear_value=avg_ear,
            head_pose=(roll, pitch, yaw),
            gaze_direction=None  # Opsiyonel
        )
        
        # Sonuçları görselleştir
        frame = drowsiness_detector.visualize(
            frame, ear_left=left_ear, ear_right=right_ear, show_metrics=True
        )
        
        # Baş duruşu ve bakış yönünü görselleştir
        if drowsiness_result['drowsiness_level'] > 0.3:  # Uykululuk seviyesi belli bir eşiği geçtiyse
            frame = mp_helper.visualize_head_pose(frame, landmarks, visualization_type='cube')
            frame, _ = mp_helper.visualize_gaze(
                frame, landmarks, ear_value=avg_ear, ear_threshold=0.21, frame_skip=2
            )
    
    # FPS hesapla
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Görüntüyü göster
    cv2.imshow("Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
mp_helper.release()
cv2.destroyAllWindows()
```

### 3. Sadece Baş Duruşu Tahmini

Yalnızca baş duruşu tahmini yaparak sürücünün baş hareketlerini izleme:

```python
from src.utils.face_landmark_detector import get_face_landmark_detector
from src.utils.head_pose_estimator import HeadPoseEstimator
import cv2
import numpy as np

# Bileşenleri oluştur
face_detector = get_face_landmark_detector()
head_pose_estimator = HeadPoseEstimator()

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Baş açıları için geçmiş verileri sakla (yumuşatma için)
pitch_history = []
yaw_history = []
roll_history = []
history_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Yüz landmarkları tespit et
    landmarks, face_detected = face_detector.detect_face_landmarks(frame)
    
    if face_detected:
        # Baş duruşunu hesapla
        rotation_matrix, angles = head_pose_estimator.calculate_head_pose(landmarks, frame)
        roll, pitch, yaw = angles
        
        # Açıları yumuşatmak için geçmiş verilere ekle
        pitch_history.append(pitch)
        yaw_history.append(yaw)
        roll_history.append(roll)
        
        # Geçmiş verileri sınırla
        if len(pitch_history) > history_size:
            pitch_history.pop(0)
            yaw_history.pop(0)
            roll_history.pop(0)
        
        # Yumuşatılmış açıları hesapla
        smooth_pitch = sum(pitch_history) / len(pitch_history)
        smooth_yaw = sum(yaw_history) / len(yaw_history)
        smooth_roll = sum(roll_history) / len(roll_history)
        
        # Görselleştir
        frame = head_pose_estimator.visualize_head_pose(
            frame, landmarks, show_axes=True, show_angles=True, visualization_type='cube'
        )
        
        # Açıları ekrana yazdır
        cv2.putText(
            frame, f"Pitch: {smooth_pitch:.1f}°", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Yaw: {smooth_yaw:.1f}°", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Roll: {smooth_roll:.1f}°", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Baş pozisyonu uyarısı
        if abs(smooth_pitch) > 20 or abs(smooth_yaw) > 30:
            cv2.putText(
                frame, "UYARI: Başınızı düz tutun!", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    
    # Görüntüyü göster
    cv2.imshow("Head Pose Estimation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
face_detector.release()
cv2.destroyAllWindows()
```

### 4. PERCLOS Hesaplama ve İzleme

Göz kapalılık süresini izleyerek PERCLOS değerini hesaplama:

```python
from src.utils import get_mediapipe_helper
from src.utils.facial_metrics import get_eye_aspect_ratio, get_perclos
import cv2
import numpy as np
import time

# MediaPipeHelper örneği oluştur
mp_helper = get_mediapipe_helper()

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# PERCLOS için gerekli değişkenler
ear_threshold = 0.21  # Göz kapalılık eşiği
eye_state_history = []  # 0: kapalı, 1: açık
fps = 30  # Tahmini FPS
window_seconds = 30  # PERCLOS hesaplama penceresi (saniye)

# Zamanlayıcı
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Yüz landmarkları tespit et
    landmarks, face_detected = mp_helper.detect_face_landmarks(frame)
    
    if face_detected:
        # Göz landmarkları al
        left_eye = mp_helper.get_eye_landmarks(landmarks, left_eye=True)
        right_eye = mp_helper.get_eye_landmarks(landmarks, left_eye=False)
        
        # EAR hesapla
        left_ear = mp_helper.get_eye_aspect_ratio(left_eye)
        right_ear = mp_helper.get_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Göz durumunu belirle
        is_eye_closed = avg_ear < ear_threshold
        eye_state_history.append(0 if is_eye_closed else 1)
        
        # Geçmiş verileri sınırla
        max_history_frames = window_seconds * fps
        if len(eye_state_history) > max_history_frames:
            eye_state_history = eye_state_history[-max_history_frames:]
        
        # PERCLOS hesapla
        perclos = get_perclos(eye_state_history, window_seconds, fps)
        
        # Görselleştir
        eye_color = (0, 0, 255) if is_eye_closed else (0, 255, 0)
        for eye in [left_eye, right_eye]:
            pts = np.array([(int(p[0]), int(p[1])) for p in eye], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, eye_color, 1)
        
        # Sonuçları göster
        cv2.putText(
            frame, f"EAR: {avg_ear:.3f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2
        )
        
        perclos_color = (0, 255, 0)  # Yeşil (normal)
        if perclos > 15:
            perclos_color = (0, 165, 255)  # Turuncu (uyarı)
        if perclos > 20:
            perclos_color = (0, 0, 255)  # Kırmızı (tehlike)
            
        cv2.putText(
            frame, f"PERCLOS: {perclos:.2f}%", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, perclos_color, 2
        )
        
        if perclos > 20:
            cv2.putText(
                frame, "UYARI: Uykululuk Tespit Edildi!", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    
    # FPS hesapla
    if frame_count % 30 == 0:  # Her 30 karede bir FPS güncelle
        end_time = time.time()
        fps = 30 / (end_time - start_time)
        start_time = end_time
    
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
    )
    
    # Görüntüyü göster
    cv2.imshow("PERCLOS Monitoring", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
mp_helper.release()
cv2.destroyAllWindows()
```

### 5. Constants Modülünü Kullanma

`constants.py` modülünü kullanarak merkezi olarak tanımlanan sabit değerleri kullanma:

```python
from src.utils import get_mediapipe_helper
from src.utils.constants import (
    EAR_THRESHOLD, 
    MAR_THRESHOLD,
    PERCLOS_WARNING_THRESHOLD,
    PERCLOS_CRITICAL_THRESHOLD,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES
)
import cv2

# MediaPipeHelper örneği oluştur
mp_helper = get_mediapipe_helper()

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Yüz landmarkları tespit et
    landmarks, face_detected = mp_helper.detect_face_landmarks(frame)
    
    if face_detected:
        # Göz landmarkları al - sabit indeksleri kullanarak
        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES if i < len(landmarks)]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES if i < len(landmarks)]
        
        # EAR hesapla
        left_ear = mp_helper.get_eye_aspect_ratio(left_eye)
        right_ear = mp_helper.get_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # MAR hesapla
        mar = mp_helper.get_mouth_aspect_ratio(landmarks)
        
        # Durumları kontrol et
        eyes_closed = avg_ear < EAR_THRESHOLD
        mouth_open = mar > MAR_THRESHOLD
        
        # Sonuçları görselleştir
        eye_status = "Kapalı" if eyes_closed else "Açık"
        mouth_status = "Açık" if mouth_open else "Kapalı"
        
        cv2.putText(
            frame, f"Gözler: {eye_status} (EAR: {avg_ear:.3f})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if eyes_closed else (0, 255, 0), 2
        )
        
        cv2.putText(
            frame, f"Ağız: {mouth_status} (MAR: {mar:.3f})", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if mouth_open else (0, 255, 0), 2
        )
    
    # Görüntüyü göster
    cv2.imshow("Facial Metrics", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
mp_helper.release()
cv2.destroyAllWindows()
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