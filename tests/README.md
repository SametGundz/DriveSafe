# ETH-XGaze Test Dokümantasyonu

Bu dizin, ETH-XGaze gaze estimation modelinin ve ilgili bileşenlerin test dosyalarını içermektedir. Bu README, testlerin nasıl çalıştırılacağını, gerekli test verilerinin nasıl hazırlanacağını ve test sonuçlarının nasıl yorumlanacağını açıklar.

## İçindekiler

1. [Test Dosyaları](#test-dosyaları)
2. [Gerekli Bağımlılıklar](#gerekli-bağımlılıklar)
3. [Test Verilerini Hazırlama](#test-verilerini-hazırlama)
4. [Testleri Çalıştırma](#testleri-çalıştırma)
5. [Test Sonuçlarını Yorumlama](#test-sonuçlarını-yorumlama)
6. [Sık Karşılaşılan Sorunlar](#sık-karşılaşılan-sorunlar)
7. [Hata Ayıklama](#hata-ayıklama)
8. [ONNX vs PyTorch Testi](#onnx-vs-pytorch-testi)

## Test Dosyaları

Bu dizinde aşağıdaki test dosyaları bulunmaktadır:

- **test_gaze_estimator.py**: GazeEstimator sınıfının test edilmesi
- **test_model_loader.py**: ModelLoader sınıfının ve model yükleme işlemlerinin test edilmesi
- **test_ear_mar.py**: Göz/Ağız açıklık oranı tespitinin test edilmesi
- **test_perclos.py**: PERCLOS (göz kapanma yüzdesi) ölçümlerinin test edilmesi
- **test_head_pose.py**: Baş duruşu tespitinin test edilmesi
- **test_drowsiness_logic.py**: Uykululuk algılama mantığının test edilmesi

## Gerekli Bağımlılıklar

Testleri çalıştırmak için aşağıdaki paketlerin yüklü olması gerekir:

```bash
pip install -r requirements.txt

# ONNX test'leri için ayrıca:
pip install onnxruntime

# GPU destekli ONNX test'leri için:
pip install onnxruntime-gpu
```

## Test Verilerini Hazırlama

Bazı testler, örnek yüz görüntülerine ihtiyaç duyar. Bu görüntüleri test etmek için:

1. `tests/data` dizinini oluşturun (eğer yoksa):
   ```bash
   mkdir -p tests/data
   ```

2. Test için bir yüz görüntüsü ekleyin:
   ```bash
   # Örnek bir yüz görüntüsü kopyalayın
   cp ornek_yuz.jpg tests/data/test_face.jpg
   ```

## Testleri Çalıştırma

### Tüm Testleri Çalıştırma

Tüm test dosyalarını çalıştırmak için:

```bash
python -m unittest discover tests
```

### Belirli Bir Test Dosyasını Çalıştırma

Belirli bir test dosyasını çalıştırmak için:

```bash
# GazeEstimator testleri
python -m unittest tests.test_gaze_estimator

# ModelLoader testleri
python -m unittest tests.test_model_loader
```

### Belirli Bir Test Metodunu Çalıştırma

Belirli bir test metodunu çalıştırmak için:

```bash
python -m unittest tests.test_gaze_estimator.TestGazeEstimator.test_preprocess_eye_image
```

### Verbose Mod

Daha detaylı test çıktısı için `-v` bayrağını kullanın:

```bash
python -m unittest discover -v tests
```

### Test Kapsamı Raporu Oluşturma

Test kapsamını kontrol etmek için `coverage` paketini kullanabilirsiniz:

```bash
# Coverage paketini yükleyin
pip install coverage

# Testleri çalıştırın ve kapsam raporu oluşturun
coverage run -m unittest discover tests
coverage report -m
```

## Test Sonuçlarını Yorumlama

- **`.`** (nokta): Test başarıyla geçti
- **`F`**: Test başarısız oldu (Failure)
- **`E`**: Test sırasında bir hata oluştu (Error)
- **`s`**: Test atlandı (Skipped)

Örnek çıktı:
```
........s.....
----------------------------------------------------------------------
Ran 15 tests in 3.245s

OK (skipped=1)
```

Bu çıktı, 14 testin başarıyla geçtiğini ve 1 testin atlandığını gösterir.

## Sık Karşılaşılan Sorunlar

### 1. ONNX Runtime Hataları

ONNX testleri sırasında şu hatayı alırsanız:
```
ImportError: No module named 'onnxruntime'
```

ONNX Runtime paketini yükleyin:
```bash
pip install onnxruntime
```

### 2. Test Görüntüsü Bulunamadı Hatası

```
Warning: Test image not found: .../tests/data/test_face.jpg
```

`tests/data` dizinine geçerli bir yüz görüntüsü ekleyin.

### 3. GPU Bulunamadı Hatası

```
CUDA device not found, falling back to CPU
```

Bu bir hata değildir. Sistem CUDA destekli bir GPU bulamadığında otomatik olarak CPU'ya geçer.

### 4. MediaPipe Hataları veya Uyarıları

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1746390552.841433  152968 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
```

Bu MediaPipe kütüphanesinin yazdığı düşük seviyeli uyarı mesajlarıdır ve testlerin çalışmasını etkilemez. Güvenle görmezden gelinebilir.

### 5. Torchvision Uyarıları

```
UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
```

Bu, Torchvision'ın daha eski API'lerini kullanmaktan kaynaklanan bir uyarıdır ve testlerin sonuçlarını etkilemez. Model sürümlerini güncelleyerek çözülebilir.

### 6. PyTorch Model Training Mode Hatası

```
AssertionError: True is not false
```

Bu hata, model test edilirken PyTorch modelinin doğru modda (eval veya train) olmamasından kaynaklanır. Varsayılan olarak, PyTorch modelleri `training=True` modunda başlatılır, ancak test sırasında genellikle `eval()` modunda olması gerekir. Şu şekilde düzeltebilirsiniz:

```python
# Hatalı kullanım - model training modunda
model = GazeResNet()  # Varsayılan olarak training=True

# Doğru kullanım - modeli eval moduna alma
model = GazeResNet()
model.eval()  # Artık model.training = False olacaktır

# Mock nesneler için training modu ayarlama
mock_model = MagicMock(spec=torch.nn.Module)
mock_model.training = False  # Doğrudan training özelliğini False olarak ayarla
```

## Hata Ayıklama

Testler sırasında hata alırsanız, şu adımları takip edebilirsiniz:

### Mock Nesneleri ile İlgili Hatalar

```
AttributeError: 'method' object has no attribute 'return_value'
```

Bu hata, bir metodun doğrudan mock'lanması yerine, onu içeren nesnenin mocklanması gerektiğini gösterir. Örneğin:

```python
# Hatalı kullanım
instance.__call__.return_value = value

# Doğru kullanım
instance = MagicMock()
instance.__call__ = MagicMock(return_value=value)
```

### Sınıf ve Örnek Değişkenleri Karışıklığı

```
AttributeError: type object 'ModelLoader' has no attribute 'base_model_dir'
```

Bu hata, sınıf düzeyinde bir özniteliğe erişmeye çalıştığınızda ancak bu özniteliğin yalnızca örneklerde mevcut olduğunda ortaya çıkar. Şu şekilde düzeltilmelidir:

```python
# Hatalı kullanım
ModelLoader.base_model_dir = value

# Doğru kullanım
loader = ModelLoader()
loader.base_model_dir = value
```

### Patch Yolları Sorunları

```
AssertionError: False is not true
```

Mock'un çağrılıp çağrılmadığını kontrol ederken, doğru import yolunun patch edildiğinden emin olun. Test edilen sınıf içinde, sınıfların nasıl import edildiğini kontrol edin. Örneğin:

```python
# GazeEstimator içinde import edildiği şekliyle patch'leyiniz
# from src.utils.model_loader import ModelLoader
with patch('src.detection.gaze_estimator.ModelLoader') as mock_loader:
    # ...
```

### PyTorch Model Modları

PyTorch modellerinin iki çalışma modu vardır:
- **Training modu**: Model eğitilirken kullanılır, gradyanlar hesaplanır ve bazı katmanlar (örn. Dropout, BatchNorm) eğitim davranışı sergiler
- **Evaluation modu**: Model tahmin yaparken kullanılır, gradyanlar hesaplanmaz ve katmanlar test davranışı sergiler

Test ederken genellikle modelleri eval moduna almalısınız. Çünkü:
1. Daha hızlı çalışır (gradyan hesaplanmaz)
2. Batch normalization katmanları sabit değerler kullanır
3. Dropout katmanları aktif olmaz (nöronları devre dışı bırakmaz)

```python
# Modeli eval moduna alma
model.eval()

# Test için
with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırakır
    output = model(input_data)
```

### MediaPipe ile İlgili Sorunlar

MediaPipe hatası alırsanız ve kamera erişimi sorunları yaşıyorsanız:

```bash
# MediaPipe'ın GPU desteğini devre dışı bırakın
export CUDA_VISIBLE_DEVICES=""
```

MediaPipe'ın çalışması için GPU erişimi gerekmiyor, bu nedenle CPU modunda çalışmaya zorlamak birçok sorunu ortadan kaldırabilir.

## ONNX vs PyTorch Testi

ETH-XGaze modelini hem ONNX hem de PyTorch formatında test etmek için özel bir test eklenmiştir. Bu test şunları doğrular:

1. ModelLoader'ın önce ONNX modelini yüklemeyi denediğini 
2. ONNX modeli yoksa PyTorch modelini yüklediğini
3. ONNX modeli için işlem süresinin daha kısa olduğunu

ONNX vs PyTorch kıyaslamasını manuel olarak gerçekleştirmek için:

```bash
# Demo uygulamasını ONNX modeli ile çalıştırma
python examples/gaze_estimation_demo.py --device cpu

# ONNX modelini geçici olarak yeniden adlandırma
mv models/eth_xgaze_model.onnx models/eth_xgaze_model.onnx.bak

# Demo uygulamasını PyTorch modeli ile çalıştırma
python examples/gaze_estimation_demo.py --device cpu

# ONNX modelini geri adlandırma
mv models/eth_xgaze_model.onnx.bak models/eth_xgaze_model.onnx
```

Demo uygulaması, kullanılan model tipini ("ONNX" veya "PyTorch") ve FPS değerini ekranda gösterecektir. 