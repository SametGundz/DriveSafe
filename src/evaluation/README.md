# Sürücü Uyku Hali Tespit Sistemi Değerlendirme Çerçevesi

Bu belge, sürücü uyku hali tespit algoritmaları için geliştirilen kapsamlı değerlendirme çerçevesinin nasıl kullanılacağını anlatır. Çerçeve, farklı tespit yöntemlerinin karşılaştırılması, bileşenlerin performans katkılarının analizi ve gerçek dünya koşullarında değerlendirme için gerekli araçları sağlar.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Kurulum ve Başlangıç](#kurulum-ve-başlangıç)
- [Modüller](#modüller)
  - [metrics.py](#metricspy) - Performans ölçüm metrikleri
  - [dataset_loader.py](#dataset_loaderpy) - Veri seti yükleme
  - [benchmark.py](#benchmarkpy) - Karşılaştırmalı değerlendirme
  - [ablation.py](#ablationpy) - Bileşen katkı analizi
  - [real_world.py](#real_worldpy) - Gerçekçi koşullarda değerlendirme
  - [visualization.py](#visualizationpy) - Sonuçların görselleştirilmesi
- [Örnek Kullanım Senaryoları](#örnek-kullanım-senaryoları)
- [Çerçeveyi Genişletme](#çerçeveyi-genişletme)

## Genel Bakış

Değerlendirme çerçevesi, sürücü uyku hali tespit algoritmalarının performansını kapsamlı ve standartlaştırılmış bir şekilde değerlendirmenizi sağlar. Ana bileşenleri:

1. **Standart Metrikler**: Doğruluk, hassasiyet, duyarlılık, F1-skoru, ROC eğrisi ve özel uyku hali metrikleri (tespit gecikmesi, yanlış alarm oranı)
2. **Veri Seti Yönetimi**: Farklı veri setleri için standart yükleme arayüzü
3. **Karşılaştırmalı Değerlendirme**: Birden fazla algoritmayı aynı koşullarda test etme
4. **Görselleştirme**: Sonuçların grafik ve tablolar ile gösterimi
5. **Ablasyon Çalışmaları**: Farklı algoritma bileşenlerinin katkılarını ölçme
6. **Gerçek Dünya Testleri**: Gerçekçi koşullar altında performans değerlendirmesi

## Kurulum ve Başlangıç

### Gereksinimler

Çerçeveyi kullanmak için aşağıdaki Python paketleri gereklidir:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python plotly seaborn
```

### Dosya Yapısı

Değerlendirme çerçevesi, `src/evaluation` dizini altında aşağıdaki modüllerden oluşur:

```
src/evaluation/
│
├── __init__.py           # Paket tanımlaması
├── metrics.py            # Performans ölçüm metrikleri
├── dataset_loader.py     # Veri seti yükleme fonksiyonları
├── benchmark.py          # Karşılaştırmalı değerlendirme
├── ablation.py           # Ablasyon çalışmaları
├── real_world.py         # Gerçek dünya değerlendirmesi
└── visualization.py      # Görselleştirme araçları
```

### Hızlı Başlangıç

Temel bir değerlendirme yapmak için:

```python
from src.evaluation.benchmark import DriversystemBenchmark

# Benchmark nesnesini oluştur
benchmark = DriversystemBenchmark(output_dir="results/benchmarks")

# Veri setini yükle
benchmark.load_dataset("path/to/dataset")

# Referans yöntemleri değerlendir
benchmark.evaluate_baseline_methods()

# Kendi yönteminizi değerlendirin
benchmark.evaluate_your_method(my_detector, "My_Method")

# Sonuçları görselleştir ve rapor oluştur
benchmark.compare_and_visualize()
benchmark.save_results()
```

## Modüller

### `metrics.py`

Uyku hali tespitinin kalitesini ölçen metrikler bu modülde yer alır.

#### Temel Kullanım

```python
from src.evaluation.metrics import calculate_basic_metrics, calculate_detection_latency

# Tahminleri ve gerçek etiketleri hazırla
y_true = [0, 1, 0, 1, 1, 0, 0, 1]  # 0: uyanık, 1: uykulu
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]  # model tahminleri

# Temel metrikleri hesapla
metrics = calculate_basic_metrics(y_true, y_pred)
print(f"Doğruluk: {metrics['accuracy']:.4f}")
print(f"Hassasiyet: {metrics['precision']:.4f}")
print(f"Duyarlılık: {metrics['recall']:.4f}")
print(f"F1-skoru: {metrics['f1_score']:.4f}")

# Tespit gecikmesini hesapla (uyku halinin ne kadar hızlı tespit edildiği)
# Gerçek uyku hali başlangıç zamanları
event_timestamps = [10.5, 45.2, 78.9]  # saniye cinsinden
# Tespit zamanları
detection_timestamps = [12.1, 46.8, 80.5]  # saniye cinsinden

avg_latency, detection_rate = calculate_detection_latency(
    event_timestamps, detection_timestamps, threshold=3.0  # 3 saniye içinde tespit edilmeli
)
print(f"Ortalama tespit gecikmesi: {avg_latency:.2f} saniye")
print(f"Tespit oranı: {detection_rate:.2f}")
```

#### Önemli Fonksiyonlar

- `calculate_basic_metrics(y_true, y_pred)`: Doğruluk, hassasiyet, duyarlılık ve F1-skoru hesaplar
- `calculate_confusion_matrix(y_true, y_pred)`: Karışıklık matrisi oluşturur
- `calculate_roc_auc(y_true, y_score)`: ROC eğrisi ve AUC değeri hesaplar
- `calculate_detection_latency(event_timestamps, detection_timestamps, threshold)`: Uyku hali tespit gecikmesini ölçer
- `calculate_false_alarm_rate(detection_timestamps, event_intervals, total_duration)`: Saatte yanlış alarm sayısını hesaplar
- `calculate_metrics(predictions, ground_truth, timestamps)`: Tüm metrikleri tek seferde hesaplar

### `dataset_loader.py`

Farklı veri setlerini standart bir arayüz ile yüklemeyi sağlar.

#### Temel Kullanım

```python
from src.evaluation.dataset_loader import load_benchmark_dataset

# Basit kullanım - veri seti tipini otomatik tespit eder
images, labels, metadata = load_benchmark_dataset('/path/to/dataset')

# Belirli veri seti tipi, bölüm ve önişleme ile yükleme
import cv2

def resize_preprocess(img):
    return cv2.resize(img, (224, 224))

images, labels, metadata = load_benchmark_dataset(
    '/path/to/dataset',
    dataset_type='nthu-ddd',  # NTHU Driver Drowsiness Dataset
    split='test',             # Test bölümü
    preprocess_fn=resize_preprocess  # Önişleme fonksiyonu
)

# Alternatif olarak, DatasetLoader sınıfını doğrudan kullanabilirsiniz
from src.evaluation.dataset_loader import get_dataset_loader

dataset = get_dataset_loader('nthu-ddd', '/path/to/dataset')
splits = dataset.get_splits()  # Mevcut veri bölümlerini al
train_data, train_labels, train_metadata = dataset.load_data(split='train')
```

#### Desteklenen Veri Setleri

- **NTHU-DDD**: NTHU Sürücü Uyku Hali Tespit Veri Seti
- **UTA-RLDD**: UTA Gerçek Yaşam Uyku Hali Veri Seti
- **Özel Veri Seti**: Standart dizin yapısına sahip özel veri setleri

Veri seti standart yapıda değilse, kendi yükleyicinizi oluşturabilirsiniz:

```python
from src.evaluation.dataset_loader import DatasetLoader

class MyCustomDataset(DatasetLoader):
    # Gerekli metodları uygulayın
    # ...

# Özel veri seti yükleyicisini kaydedin
get_dataset_loader.register('my-dataset', MyCustomDataset)
```

### `benchmark.py`

Farklı uyku hali tespit algoritmalarını karşılaştırmak ve analiz etmek için en önemli modüldür. `DriversystemBenchmark` sınıfı, tespit yöntemlerini değerlendirmek, sonuçları görselleştirmek ve karşılaştırmalı raporlar üretmek için kapsamlı bir arayüz sağlar.

#### Temel Kullanım

```python
from src.evaluation.benchmark import DriversystemBenchmark

# 1. Benchmark nesnesini oluşturma
benchmark = DriversystemBenchmark(
    output_dir="results/driver_benchmark",  # Sonuçların kaydedileceği dizin
    cache_dir="results/cache"               # Önbellek dizini
)

# 2. Veri setini yükleme
dataset, labels, metadata = benchmark.load_dataset(
    dataset_path="datasets/drowsiness_dataset",
    split="test",                  # Test bölümünü kullan
    dataset_type=None,             # Otomatik tespit (veya 'nthu-ddd', 'uta-rldd', 'custom')
    preprocess_fn=lambda img: cv2.resize(img, (224, 224))  # Önişleme (isteğe bağlı)
)
print(f"Veri seti yüklendi: {len(dataset)} görüntü, {sum(labels)} uyku hali örneği")

# 3. Temel referans yöntemlerini değerlendirme
baseline_results = benchmark.evaluate_baseline_methods()
print("Temel yöntemler değerlendirildi:")
for method, metrics in baseline_results.items():
    print(f"  - {method}: Doğruluk = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")

# 4. Kendi yönteminizi değerlendirme
def my_drowsiness_detector(image):
    # Görüntüyü işle ve sınıflandır
    # Burada kendi algoritmanız olacak
    return 1 if is_drowsy(image) else 0  # 1: uykulu, 0: uyanık

# Özel yöntem değerlendirmesi
custom_metrics = benchmark.evaluate_your_method(
    custom_detector=my_drowsiness_detector,
    method_name="My_Advanced_Method",
    batch_size=1  # İsterseniz toplu işleme kullanabilirsiniz (batch_size > 1)
)
print(f"Özel yöntem değerlendirildi: Doğruluk = {custom_metrics['accuracy']:.4f}")

# 5. Sonuçları görselleştirme ve rapor oluşturma
benchmark.compare_and_visualize(
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'false_alarm_rate'],
    output_prefix="drowsiness_comparison"
)

# 6. Sonuçları kaydetme
results_file = benchmark.save_results("final_benchmark_results.json")
```

#### Görselleştirme (Yeni Özellik)

```python
# Kapsamlı görselleştirmeler oluşturma
vis_paths = benchmark.visualize_results(
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 
             'avg_processing_time', 'false_alarm_rate'],
    output_dir="thesis/figures",
    prefix="drowsiness_detection",
    include_interactive=True  # HTML tabanlı etkileşimli grafikler
)

# Oluşturulan görselleştirmelerin listesi
for viz_type, path in vis_paths.items():
    print(f"- {viz_type}: {path}")
```

#### Karşılaştırma Tabloları Oluşturma

Tez veya makale için Markdown veya CSV formatında karşılaştırma tabloları oluşturabilirsiniz:

```python
# 1. Doğrudan DriversystemBenchmark ile
markdown_table = benchmark.generate_comparison_table(
    export_csv=True,                # CSV dosyası olarak da kaydet
    metrics=[                       # Karşılaştırılacak metrikler
        'accuracy', 'precision', 'recall', 'f1_score', 
        'avg_processing_time', 'false_alarm_rate'
    ],
    include_platform_info=True      # Platform gereksinimlerini de göster
)

# 2. Bağımsız fonksiyon ile (kaydedilmiş sonuçlarla)
from src.evaluation.benchmark import generate_comparison_table
import json

with open("results/benchmark_results.json", "r") as f:
    saved_results = json.load(f)

table = generate_comparison_table(
    saved_results,
    export_csv=True,
    output_dir="thesis/tables"
)

# Tabloyu tez belgenize ekleme
with open("thesis/drowsiness_methods_comparison.md", "w") as f:
    f.write(markdown_table)
```

#### Kaydedilmiş Sonuçları Yükleme

```python
# Önceki benchmark sonuçlarını yükleyip analiz etme
new_benchmark = DriversystemBenchmark()
new_benchmark.load_results("results/driver_benchmark/previous_results.json")

# Yüklenen sonuçları görselleştirme
new_benchmark.compare_and_visualize()
```

### `ablation.py`

Ablasyon çalışmaları, bir algoritmanın farklı bileşenlerinin genel performansa etkisini ölçmek için kullanılır. Bu modül, sistematik olarak farklı bileşenleri devre dışı bırakarak veya değiştirerek her birinin katkısını analiz etmenizi sağlar.

#### Temel Kullanım

```python
from src.evaluation.ablation import AblationStudy

# 1. Temel model fonksiyonu (tüm bileşenler etkin)
def full_model(image, component_flags=None, **kwargs):
    # component_flags: {'bileşen_adı': True/False} şeklinde sözlük
    
    # Eğer bileşen bayrakları belirtilmişse, bunları kontrol et
    use_eye_detector = True
    use_head_pose = True
    use_blink_analyzer = True
    
    if component_flags:
        if 'eye_detector' in component_flags:
            use_eye_detector = component_flags['eye_detector']
        if 'head_pose' in component_flags:
            use_head_pose = component_flags['head_pose']
        if 'blink_analyzer' in component_flags:
            use_blink_analyzer = component_flags['blink_analyzer']
    
    # Algoritmanın gerçek uygulaması burada...
    # Bileşenleri etkinleştir/devre dışı bırak
    
    # Örnek sonuç
    return 1 if predict_drowsy(image) else 0  # 1: uykulu, 0: uyanık

# 2. Ablasyon çalışması oluştur
ablation = AblationStudy(
    base_model_fn=full_model,
    output_dir="results/ablation_study"
)

# 3. İncelenecek bileşenleri kaydet
ablation.register_component("eye_detector")     # Göz dedektörü
ablation.register_component("head_pose")        # Kafa pozisyonu
ablation.register_component("blink_analyzer")   # Göz kırpma analizcisi

# 4. Özel varyantları kaydet (opsiyonel)
def fast_model_variant(image):
    # Daha hızlı çalışan model varyantı
    # Örneğin, belirli bileşenleri basitleştirebilir
    return process_faster(image)

# Hızlı varyantı ekle (göz kırpma analizcisi devre dışı)
ablation.register_variant(
    "fast_variant",
    {"blink_analyzer": False},  # Bileşen konfigürasyonu
    fast_model_variant          # Varyant fonksiyonu
)

# 5. Veri seti üzerinde değerlendirme yap
results = ablation.evaluate_on_dataset(
    dataset_name="nthu-ddd",
    dataset_path="datasets/nthu_ddd",
    split="test"
)

# 6. Sonuçları görselleştirme ve rapor oluşturma
# Bileşen etkilerini görselleştir
ablation.plot_component_impact(
    metric="f1_score",
    output_file="results/ablation_study/component_impact.png"
)

# HTML raporu oluştur
report_path = ablation.generate_report()

# 7. Sonuçları kaydet
results_file = ablation.save_results("ablation_results.json")
```

#### Konfigürasyon Dosyası ile Ablasyon (Daha Kolay Yöntem)

Daha basit ve esnek bir yaklaşım için konfigürasyon dosyası kullanabilirsiniz:

```python
from src.evaluation.ablation import perform_ablation_study

# 1. Konfigürasyon dosyası oluştur
import json

config = {
    "base_model": {
        "name": "DrowsinessDetector",  # Model sınıf adı
        "params": {                    # Model parametreleri
            "threshold": 0.5,
            "use_gpu": True
        }
    },
    "components": [                    # İncelenecek bileşenler
        "eye_detector",
        "head_pose",
        "perclos_calculator",
        "blink_frequency_analyzer"
    ],
    "dataset": {
        "name": "nthu-ddd",
        "split": "test"
    },
    "output": {
        "save_results": True,
        "generate_report": True,
        "plot_metrics": ["accuracy", "f1_score"]
    }
}

with open("configs/ablation_config.json", "w") as f:
    json.dump(config, f, indent=4)

# 2. Ablasyon çalışmasını çalıştır
results = perform_ablation_study(
    "configs/ablation_config.json",
    "datasets/nthu_ddd",
    output_dir="results/ablation_study"
)

# 3. Sonuçları analiz et
summary = results["summary"]
print("Ablasyon çalışması sonuçları:")
print(f"Temel model (tüm bileşenler): F1 = {summary['configurations']['base']['metrics']['f1_score']:.4f}")

for component in config["components"]:
    comp_key = f"no_{component}"
    if comp_key in summary["comparative_metrics"]:
        impact = summary["comparative_metrics"][comp_key]["f1_score"]["relative"] * 100
        print(f"- {component} olmadan: F1 değişimi = {impact:.2f}%")
```

#### Tek Bir Model için Hızlı Değerlendirme

```python
from src.evaluation.ablation import evaluate_on_dataset

# Model sınıfını oluştur
model = DrowsinessDetector(threshold=0.5)

# Tek bir değerlendirme yap
eval_results = evaluate_on_dataset(
    model=model,
    dataset_path="datasets/nthu_ddd",
    dataset_type="nthu-ddd",
    split="test",
    batch_size=32
)

# Sonuçları görüntüle
print(f"Model performansı:")
print(f"Doğruluk: {eval_results['metrics']['accuracy']:.4f}")
print(f"F1 Skoru: {eval_results['metrics']['f1_score']:.4f}")
print(f"Yanlış alarm oranı: {eval_results['metrics']['false_alarm_rate']:.2f} alarm/saat")
```

### `real_world.py`

Tespit yöntemlerini gerçekçi koşullar altında değerlendirmek için araçlar:

- Çeşitli koşullar altında video tabanlı değerlendirme
- Zorlu ortamların simülasyonu:
  - Farklı aydınlatma koşulları
  - Hareket bulanıklığı ve gürültü
  - İşleme gecikmeleri
- Gerçek dünya metrikleri:
  - İşleme süresi ve gerçek zamanlı faktör
  - Tespit gecikmesi
  - Yanlış alarm oranları
- Sağlamlık testi için çoklu senaryo değerlendirmesi
- **Yeni:** Simüle edilmiş gerçek sürüş koşullarında kapsamlı değerlendirme
- **Yeni:** Sürücü demografik faktörlerine (gözlük, etnik köken) göre performans analizi
- **Yeni:** Farklı aydınlatma koşullarında ve sürücü baş pozisyonlarında değerlendirme

```python
from src.evaluation.real_world import RealWorldEvaluator, create_scenario

# Değerlendiriciyi başlat
evaluator = RealWorldEvaluator(output_dir="results/real_world")

# Test senaryosu oluştur
scenario = create_scenario(
    name="gece_surusu",
    description="Gece sürüş koşulları altında değerlendirme",
    videos=["video1.mp4", "video2.mp4"],
    conditions=[
        {"name": "loş_ışık", "runtime_conditions": {"lighting": "dim"}},
        {"name": "hareket", "runtime_conditions": {"blur": "motion"}}
    ]
)

# Senaryoda yöntem değerlendir
evaluator.evaluate_on_scenario(yontem_fn, scenario, method_name="yontem1")

# Birden çok yöntemi karşılaştır
methods = {
    "yontem1": yontem1_fn,
    "yontem2": yontem2_fn
}
evaluator.evaluate_multiple_methods(methods, scenario)

# Rapor oluştur
evaluator.save_results()
evaluator.generate_report()
```

#### Gerçek Sürüş Koşullarında Simülasyon Değerlendirmesi (Yeni Özellik)

```python
from src.evaluation.real_world import evaluate_in_real_conditions

# Uzun süreli gerçek sürüş simülasyonu
sonuclar = evaluate_in_real_conditions(
    method_fn=tespit_yontemi,
    duration_minutes=60,  # 1 saatlik simülasyon
    output_dir="results/surucu_simulasyonu",
    method_name="tespit_sistemi_v2"
)

# Farklı demografik faktörlerde performans analizi
etnik_sonuclar = sonuclar["results_by_condition"]["ethnicity"]
print("\nEtnik kökene göre tespit performansı:")
for etnik, metrikler in etnik_sonuclar.items():
    print(f"{etnik}: Tespit oranı: {metrikler['detection_rate']:.4f}")

# Farklı aydınlatma koşullarında performans analizi  
aydinlatma_sonuclar = sonuclar["results_by_condition"]["lighting"]
print("\nAydınlatma koşullarına göre tespit performansı:")
for aydinlatma, metrikler in aydinlatma_sonuclar.items():
    print(f"{aydinlatma}: Tespit oranı: {metrikler['detection_rate']:.4f}")

# Özet metrikleri görüntüle
print("\nÖzet Metrikler:")
print(f"Ortalama tespit oranı: {sonuclar['metrics']['detection_rate']:.4f}")
print(f"Tespit gecikmesi: {sonuclar['metrics']['detection_latency']:.2f} saniye")
print(f"Yanlış alarm oranı: {sonuclar['metrics']['false_alarm_rate']:.2f} alarm/saat")
```

### `visualization.py` (Yeni)

Değerlendirme sonuçlarını görselleştiren fonksiyonlar içerir. Bu modül, karşılaştırmalı değerlendirme sonuçlarınızı anlaşılır ve etkileyici grafikler, çizelgeler ve diyagramlar halinde sunmanıza olanak tanır.

#### Temel Kullanım

```python
from src.evaluation.visualization import (
    plot_metric_comparison,
    plot_roc_curves, 
    plot_processing_time_comparison,
    plot_metric_vs_time,
    generate_visualization_set
)

# Sonuçları önceden oluşturulan benchmark'tan alın
# veya JSON dosyasından yükleyin
import json
with open("results/benchmark_results.json", "r") as f:
    results = json.load(f)

# 1. Tek bir metrik için çubuk grafiği oluşturma
fig = plot_metric_comparison(
    results,
    metric='accuracy',  # Doğruluk metriğini kullan
    output_file='figures/accuracy_comparison.png',
    title='Uyku Hali Tespit Yöntemleri Doğruluk Karşılaştırması'
)

# 2. ROC eğrileri oluşturma (tüm yöntemler için)
fig = plot_roc_curves(
    results,
    output_file='figures/roc_curves.png'
)

# 3. İşlem süresini karşılaştırma
fig = plot_processing_time_comparison(
    results,
    output_file='figures/processing_times.png',
    metric='avg_processing_time'  # Kullanılacak zaman metriği
)

# 4. Doğruluk ve işlem süresi ilişkisini gösteren dağılım grafiği
fig = plot_metric_vs_time(
    results,
    perf_metric='accuracy',  # Y ekseni: doğruluk
    time_metric='avg_processing_time',  # X ekseni: işlem süresi
    output_file='figures/accuracy_vs_time.png'
)

# 5. Tüm görselleştirmeleri tek seferde oluşturma
vis_files = generate_visualization_set(
    results,
    output_dir='thesis/figures',
    prefix='drowsiness_methods'
)

# Oluşturulan dosyaların listesi
for vis_type, file_path in vis_files.items():
    print(f"{vis_type}: {file_path}")
```

#### Önemli Fonksiyonlar

1. **Temel Karşılaştırmalar**:
   - `plot_metric_comparison()`: Farklı yöntemleri belirli bir metriğe göre karşılaştıran çubuk grafiği
   - `plot_metrics_by_dataset()`: Farklı veri setlerinde performans karşılaştırması
   - `plot_roc_curves()`: Tüm yöntemler için ROC eğrileri

2. **Zaman/Performans Analizleri**:
   - `plot_processing_time_comparison()`: İşlem sürelerini karşılaştırma
   - `plot_metric_vs_time()`: Performans-zaman ilişkisini gösteren dağılım grafiği
   - `plot_detection_latency_comparison()`: Tespit gecikmelerini karşılaştırma

3. **Koşullu Analizler**:
   - `plot_condition_comparison()`: Farklı koşullara göre performans karşılaştırması
   - `plot_false_alarm_comparison()`: Yanlış alarm oranlarını karşılaştırma

4. **Toplu Görselleştirme**:
   - `generate_visualization_set()`: Tüm ilgili görselleştirmeleri tek seferde oluşturma

### `real_world.py`

Gerçek dünya koşullarında tespit yöntemlerinin değerlendirilmesi için fonksiyonlar içerir. Bu modül, video akışı üzerinde değerlendirme, farklı ortam koşullarını simülasyon ve kapsamlı gerçek dünya testleri yapmanızı sağlar.

#### Temel Kullanım

```python
from src.evaluation.real_world import RealWorldEvaluator, create_scenario

# 1. Değerlendirici oluşturma
evaluator = RealWorldEvaluator(output_dir="results/real_world_tests")

# 2. Video üzerinde değerlendirme
video_results = evaluator.evaluate_on_video(
    method_fn=my_detection_method,
    video_path="test_videos/highway_drive.mp4",
    ground_truth=None,  # İsteğe bağlı ground truth verileri
    method_name="Advanced_Method",
    process_rate=1.0,  # Her kareyi işle (0.5: her ikinci kare)
    # Farklı ortam koşullarını simüle et
    runtime_conditions={
        "lighting": "dim",        # loş ışık koşulları
        "blur": 2,                # hafif bulanıklık
        "noise": "salt_pepper"    # gürültü ekleme
    }
)

# 3. Senaryo oluşturma ve değerlendirme
test_scenario = create_scenario(
    name="mixed_lighting_conditions",
    description="Farklı aydınlatma koşullarında değerlendirme",
    videos=["test_videos/daylight.mp4", "test_videos/night.mp4", "test_videos/tunnel.mp4"],
    conditions=[
        {"name": "parlak_isik", "runtime_conditions": {"lighting": "bright"}},
        {"name": "los_isik", "runtime_conditions": {"lighting": "dim"}},
        {"name": "karisik_isik", "runtime_conditions": {"lighting": "mixed"}}
    ]
)

# Senaryoda yöntemi değerlendir
scenario_results = evaluator.evaluate_on_scenario(
    method_fn=my_detection_method,
    scenario_config=test_scenario,
    method_name="Advanced_Method"
)

# 4. Birden fazla yöntemi karşılaştırma
methods = {
    "Method_A": method_a_fn,
    "Method_B": method_b_fn,
    "Baseline": baseline_method_fn
}

comparison_results = evaluator.evaluate_multiple_methods(
    methods=methods,
    scenario_config=test_scenario
)

# 5. Rapor oluşturma ve sonuçları kaydetme
evaluator.save_results("real_world_evaluation.json")
report_path = evaluator.generate_report("real_world_report.html")
```

#### Kapsamlı Gerçek Sürüş Simülasyonu

Bu özellik, kapsamlı bir sürüş deneyimini simüle ederek algoritmanızın farklı koşullarda nasıl performans göstereceğini değerlendirir:

```python
from src.evaluation.real_world import evaluate_in_real_conditions

# Simülasyon başlatma
results = evaluate_in_real_conditions(
    method_fn=my_detection_method,
    duration_minutes=30,              # 30 dakikalık simülasyon
    output_dir="results/simulation",
    # Demografik faktörleri dahil et
    include_glasses=True,             # Gözlüklü ve gözlüksüz sürücüler
    include_ethnicities=True,         # Farklı etnik kökenler
    include_orientations=True,        # Farklı yüz oryantasyonları
    method_name="Advanced_Algorithm"
)

# Demografik faktörlere göre performans analizi
print("\n== Demografik Faktörlere Göre Performans ==")

# Gözlük durumuna göre
glasses_results = results["results_by_condition"]["glasses"]
for condition, metrics in glasses_results.items():
    print(f"Gözlük durumu: {condition}")
    print(f"  - Tespit oranı: {metrics['detection_rate']:.4f}")
    print(f"  - Yanlış alarm: {metrics['false_alarm_rate']:.2f}/saat")

# Yüz oryantasyonuna göre
orientation_results = results["results_by_condition"]["orientation"]
for orientation, metrics in orientation_results.items():
    print(f"Yüz oryantasyonu: {orientation}")
    print(f"  - Tespit oranı: {metrics['detection_rate']:.4f}")
    print(f"  - Tespit gecikmesi: {metrics['detection_latency']:.2f} saniye")

# Aydınlatma koşullarına göre
lighting_results = results["results_by_condition"]["lighting"]
for lighting, metrics in lighting_results.items():
    print(f"Aydınlatma: {lighting}")
    print(f"  - Tespit oranı: {metrics['detection_rate']:.4f}")
    print(f"  - Ortalama işlem süresi: {metrics['avg_processing_time']:.4f} saniye")

# Özet sonuçlar
print("\n== Özet Performans Metrikleri ==")
print(f"Genel tespit oranı: {results['metrics']['detection_rate']:.4f}")
print(f"Ortalama tespit gecikmesi: {results['metrics']['detection_latency']:.2f} saniye")
print(f"Yanlış alarm oranı: {results['metrics']['false_alarm_rate']:.2f} alarm/saat")
print(f"Gerçek zamanlı faktör: {results['metrics']['realtime_factor']:.2f}x")
```

#### Simüle Edilen Senaryolar

`evaluate_in_real_conditions` fonksiyonu aşağıdaki senaryo ve koşulları simüle eder:

1. **Yol Senaryoları**:
   - Otoyol sürüşü (uzun, monoton yolculuklar)
   - Şehir içi sürüş (dur-kalk trafik)
   - Kırsal yol (az trafik, monoton)
   - Trafik sıkışıklığı (uzun bekleme süreleri)

2. **Çevresel Koşullar**:
   - Aydınlatma varyasyonları (gündüz, gece, tünel, gün batımı)
   - Hareket bulanıklığı (farklı derecelerde)
   - Gürültü ve parazit 

3. **Sürücü Faktörleri**:
   - Gözlük kullanımı
   - Farklı yüz oryantasyonları
   - Farklı etnik kökenler (yüz özellikleri)

## Örnek Kullanım Senaryoları

### Senaryo 1: Yeni Bir Algoritma Değerlendirme ve Karşılaştırma

Geliştirdiğiniz yeni bir uyku hali tespit algoritmasını değerlendirmek ve mevcut yöntemlerle karşılaştırmak istediğinizi varsayalım.

```python
# 1. İlgili modülleri içe aktarın
from src.evaluation.benchmark import DriversystemBenchmark
from src.detection.my_algorithm import MyDetector

# 2. Benchmark nesnesini başlatın
benchmark = DriversystemBenchmark(output_dir="results/algorithm_comparison")

# 3. Test veri setini yükleyin
benchmark.load_dataset(
    dataset_path="datasets/nthu_ddd",
    split="test",
    preprocess_fn=lambda img: cv2.resize(img, (224, 224))
)

# 4. Referans yöntemleri değerlendirin
baseline_results = benchmark.evaluate_baseline_methods()

# 5. Kendi algoritmanızı değerlendirin
my_detector = MyDetector(threshold=0.6)
benchmark.evaluate_your_method(my_detector, "My_Advanced_Algorithm")

# 6. Sonuçları karşılaştırın ve görselleştirin
benchmark.compare_and_visualize()

# 7. Akademik makale veya tez için karşılaştırma tablosu oluşturun
table = benchmark.generate_comparison_table(
    export_csv=True,
    metrics=['accuracy', 'f1_score', 'false_alarm_rate', 'avg_processing_time']
)

# 8. Sonuçları kaydedin
benchmark.save_results("algorithm_comparison.json")
```

### Senaryo 2: Algoritma Bileşenlerinin Etkisini Analiz Etme

Algoritmanızın farklı bileşenlerinin genel performansa katkısını ölçmek için:

```python
# 1. İlgili modülleri içe aktarın
from src.evaluation.ablation import perform_ablation_study

# 2. Konfigürasyon dosyasını oluşturun
import json

config = {
    "base_model": {
        "name": "MyDetector",  # Algoritmanızın sınıf adı
        "params": {
            "threshold": 0.5,
            "use_cnn": True
        }
    },
    "components": [
        "eye_detector",
        "head_pose_estimator",
        "blink_analyzer",
        "yawning_detector"
    ],
    "dataset": {
        "name": "nthu-ddd",
        "split": "test"
    },
    "output": {
        "save_results": True,
        "generate_report": True,
        "plot_metrics": ["accuracy", "f1_score"]
    }
}

with open("configs/my_ablation_config.json", "w") as f:
    json.dump(config, f, indent=4)

# 3. Ablasyon çalışmasını çalıştırın
results = perform_ablation_study(
    "configs/my_ablation_config.json",
    "datasets/nthu_ddd",
    output_dir="results/component_analysis"
)

# 4. Sonuçları analiz edin
print("\n== Bileşen Analizi Sonuçları ==")
print(f"Temel model performansı: {results['summary']['configurations']['base']['metrics']['f1_score']:.4f} F1 skoru")

for component in config["components"]:
    if f"no_{component}" in results["summary"]["comparative_metrics"]:
        impact = results["summary"]["comparative_metrics"][f"no_{component}"]["f1_score"]["relative"] * 100
        print(f"- {component}: Etkisi = {impact:.2f}% (F1 skorundaki göreceli değişim)")
```

### Senaryo 3: Gerçek Dünya Koşullarında Dayanıklılık Testi

Algoritmanızın gerçek dünya koşullarındaki dayanıklılığını test etmek için:

```python
# 1. İlgili modülleri içe aktarın
from src.evaluation.real_world import evaluate_in_real_conditions, RealWorldEvaluator
from src.detection.my_algorithm import MyDetector

# 2. Algoritmanızı oluşturun
my_detector = MyDetector(threshold=0.5)
detector_func = lambda img: my_detector.predict(img)

# 3. Kapsamlı simülasyon değerlendirmesi yapın
simulation_results = evaluate_in_real_conditions(
    method_fn=detector_func,
    duration_minutes=60,  # 1 saatlik simülasyon
    output_dir="results/real_world_simulation",
    include_glasses=True,
    include_ethnicities=True,
    include_orientations=True,
    method_name="MyDetector_v2"
)

# 4. Video tabanlı değerlendirme
evaluator = RealWorldEvaluator(output_dir="results/video_evaluation")

# Test videolarınızı değerlendirin
videos = [
    "test_videos/highway_day.mp4",
    "test_videos/highway_night.mp4",
    "test_videos/city_traffic.mp4"
]

for video in videos:
    video_name = video.split("/")[-1].split(".")[0]
    results = evaluator.evaluate_on_video(
        method_fn=detector_func,
        video_path=video,
        method_name=f"MyDetector_{video_name}"
    )
    
    print(f"\nVideo: {video_name}")
    print(f"Tespit oranı: {results['metrics']['detection_rate']:.4f}")
    print(f"Yanlış alarm oranı: {results['metrics']['false_alarm_rate']:.2f}/saat")
    print(f"Ortalama işlem süresi: {results['metrics']['avg_processing_time']:.4f} saniye")

# 5. Rapor oluşturun
evaluator.generate_report("real_world_evaluation_report.html")
```

## Çerçeveyi Genişletme

### Yeni Metrikler Ekleme

Özel metrikler eklemek için `metrics.py` dosyasını genişletebilirsiniz:

```python
# src/evaluation/metrics.py dosyasına ekleyin
def calculate_my_custom_metric(y_true, y_pred, **params):
    """
    Özel bir performans metriği hesaplar.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        **params: Ek parametreler
        
    Returns:
        float: Metrik değeri
    """
    # Metrik hesaplamanızı burada uygulayın
    # Örnek: y_true ve y_pred'in bir kombinasyonu
    importance_weight = params.get('importance_weight', 1.0)
    
    # Örnek metrik hesaplama
    # Bu basit bir örnektir, kendi metriğinizi tasarlayın
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    
    # Ağırlıklı doğruluk
    weighted_accuracy = (correct / total) * importance_weight
    
    return weighted_accuracy
```

Yeni metriği `DriversystemBenchmark` sınıfıyla kullanmak için:

```python
from src.evaluation.metrics import calculate_my_custom_metric

# Özel metriği çağırma
def evaluate_method_with_custom_metric(benchmark, detector, method_name):
    # Standart değerlendirme yapın
    standard_metrics = benchmark.evaluate_your_method(detector, method_name)
    
    # Özel metriği ekleyin
    y_true = benchmark.labels
    y_pred = standard_metrics['predictions']
    
    custom_metric = calculate_my_custom_metric(
        y_true, y_pred, importance_weight=1.5
    )
    
    # Sonuçlara ekleyin
    benchmark.results[method_name]['custom_metric'] = custom_metric
    
    return benchmark.results[method_name]
```

### Yeni Veri Seti Desteği Ekleme

Eğer sistemin desteklemediği bir veri seti kullanmak istiyorsanız, özel bir yükleyici sınıfı oluşturabilirsiniz:

```python
from src.evaluation.dataset_loader import DatasetLoader

class MyCustomDataset(DatasetLoader):
    """Özel veri seti formatı için yükleyici."""
    
    def __init__(self, dataset_path, cache_dir=None):
        super().__init__(dataset_path, cache_dir)
        self._metadata = None
    
    def load_metadata(self):
        """Veri seti meta verilerini yükler."""
        if self._metadata is not None:
            return self._metadata
        
        # Meta verileri oluşturun veya yükleyin
        self._metadata = {
            "subjects": [...],  # Veri setindeki özneler/kişiler
            "conditions": [...],  # Koşullar
            "splits": {
                "train": [...],  # Eğitim bölümü
                "validation": [...],  # Doğrulama bölümü
                "test": [...]  # Test bölümü
            }
        }
        
        return self._metadata
    
    def get_splits(self):
        """Kullanılabilir veri bölümlerini döndürür."""
        metadata = self.load_metadata()
        return list(metadata["splits"].keys())
    
    def load_data(self, split='train', subset=None):
        """Veri setini yükler."""
        metadata = self.load_metadata()
        
        # Belirtilen bölüm için görüntü yollarını, etiketleri ve meta verileri toplayın
        X = []  # Görüntü yolları
        y = []  # Etiketler (0: uyanık, 1: uykulu)
        sample_metadata = []  # Örnek meta verileri
        
        # Veri seti formatınıza uygun yükleme kodunu buraya yazın
        # ...
        
        return X, np.array(y), sample_metadata
```

Özel yükleyiciyi kaydetmek ve kullanmak için:

```python
from src.evaluation.dataset_loader import get_dataset_loader

# Özel veri seti yükleyicisini kaydedin
get_dataset_loader.register_loader('my-custom-dataset', MyCustomDataset)

# Artık diğer veri setleri gibi kullanabilirsiniz
dataset = get_dataset_loader('my-custom-dataset', 'path/to/my_dataset')
splits = dataset.get_splits()
X, y, metadata = dataset.load_data(split='test')
```

## Son Güncellemeler ve İpuçları

- **Önbellek Kullanımı**: Büyük veri setleri için `cache_dir` parametresini kullanarak verileri önbelleğe alın
- **Görselleştirme Kaydetme**: Tüm görselleştirmeleri `output_dir` parametresiyle belirli bir dizine kaydedin
- **Batch İşleme**: Büyük veri setleri için `batch_size` parametresiyle toplu işleme yapın
- **Sonuçları Kaydetme**: Her zaman `save_results()` ile sonuçları kaydedin, böylece daha sonra analiz için yükleyebilirsiniz
- **HTML Raporları**: Kapsamlı görselleştirmeler ve analizler için `generate_report()` kullanın

Son güncelleme: 2025-05-05 