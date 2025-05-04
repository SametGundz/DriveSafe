# Gaze Estimation Models

Bu dizin, sürücü uykululuk tespit sistemi tarafından kullanılan gaze estimation (bakış tahmini) modellerini içerir.

## ETH-XGaze Model

ETH-XGaze, çeşitli baş duruşları altında değişen bakış açılarını doğru bir şekilde tahmin etmek için geliştirilmiş son teknoloji bir bakış tahmini metodolojisidir.

### Desteklenen Model Formatları

Sistemimiz iki farklı formatta ETH-XGaze modelini desteklemektedir:

1. **ONNX Formatı (Tavsiye Edilen)**: 
   - Dosya adı: `eth_xgaze_model.onnx`
   - ONNX (Open Neural Network Exchange) formatı çoklu platform desteği ve daha hızlı çalışma süresi sağlar
   - Gerçek zamanlı uygulamalar için en uygun seçenektir

2. **PyTorch Formatı**:
   - Dosya adı: `eth_xgaze_model.pth`
   - PyTorch formatı daha esnek olabilir ancak genellikle ONNX'ten daha yavaş çalışır

Sistem öncelikle ONNX modeli arar, bulamazsa PyTorch modelini kullanır.

### ONNX Modelinin Avantajları

- **Daha hızlı çalışma süresi**: ONNX, derin öğrenme modellerini optimize ederek daha hızlı çalışmasını sağlar
- **Hafıza verimliliği**: Daha az bellek kullanımı
- **Çoklu platform desteği**: Farklı donanım ve platformlarda kullanılabilir
- **Mobil cihaz uyumluluğu**: Düşük güçlü cihazlarda bile verimli çalışır

### Model Dosyalarını Yerleştirme

Elinizdeki modeli kullanmak için:

- ONNX modeli için: `eth_xgaze.onnx` dosyasını `models/eth_xgaze_model.onnx` olarak yerleştirin
- PyTorch modeli için: `eth_xgaze.pth.tar` dosyasını `models/eth_xgaze_model.pth` olarak yerleştirin

### Bağımlılıklar

ONNX modeli kullanmak için aşağıdaki Python paketini yüklemeniz gerekir:

```bash
pip install onnxruntime
```

GPU hızlandırması için (isteğe bağlı):

```bash
pip install onnxruntime-gpu
```

### Model İndirme

ETH-XGaze modelini indirmek için:

1. ETH-XGaze projesinin [resmi web sitesini](https://xgaze.eyecom-lab.com) ziyaret edin.
2. Model almak için gerekli kayıt işlemlerini tamamlayın.
3. İndirilen modeli uygun formatta bu dizine yerleştirin.

**Not:** Model dosyasını bulamazsanız, sistem otomatik olarak aynı mimaride bir model oluşturacak, ancak bu model eğitilmemiş olacağından doğru sonuçlar vermeyecektir.

### Model Alternatifi

Eğer ETH-XGaze modelinin önceden eğitilmiş ağırlıklarını elde edemezseniz, aşağıdaki alternatiflerden birini kullanabilirsiniz:

1. Kendiniz eğitin: ETH-XGaze GitHub reposundaki eğitim kodunu kullanarak kendi modelinizi eğitebilirsiniz.
2. PyTorch Hub'dan bir model kullanın: Daha basit bakış tahmini modelleri için PyTorch Hub'ı kullanabilirsiniz.

### Kamera Kalibrasyonu

Optimum performans için, `config/config.yaml` dosyasındaki kamera kalibrasyon parametrelerini kendi kameranıza göre güncellemelisiniz. Kalibrasyon parametre değerlerini almak için OpenCV'nin kamera kalibrasyon araçlarını kullanabilirsiniz. 