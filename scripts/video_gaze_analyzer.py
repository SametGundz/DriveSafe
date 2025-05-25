#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Gaze Zone Analyzer

Bu modül, video kayıtlarını yükleyip gaze zone analizi yaparak raporlar oluşturur.
"""

import os
import cv2
import json
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Gaze zone analiz modüllerini import et
# Bu modüller sizin gaze zone analiz kodunuza göre değişecektir
try:
    from video_frame import VideoFrameProcessor
except ImportError:
    print("Warning: VideoFrameProcessor modülü bulunamadı.")
    
# Evaulation GUI'dan ScientificGazeEvaluator'ı import et
try:
    from scripts.evaulation_gui import ScientificGazeEvaluator
except ImportError:
    try:
        from evaulation_gui import ScientificGazeEvaluator
    except ImportError:
        print("Warning: ScientificGazeEvaluator modülü bulunamadı.")


class VideoGazeAnalyzer:
    """Video kayıtlarını yükleyip gaze zone analizi yaparak raporlar oluşturan sınıf."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Video analiz sınıfını başlat.
        
        Args:
            model_path: Gaze zone analiz modelinin yolu (opsiyonel)
        """
        self.model_path = model_path
        self.video_processor = None
        self.current_video_path = None
        self.frame_count = 0
        self.fps = 0
        self.duration = 0
        self.width = 0
        self.height = 0
        
        # Analiz sonuçları
        self.frame_results = {}
        self.predictions = {}
        self.ground_truth = {}
        
        # Modeli yükle
        self._load_model()
    
    def _load_model(self):
        """Gaze zone analiz modelini yükle."""
        try:
            # Eğer VideoFrameProcessor sınıfı mevcutsa
            if 'VideoFrameProcessor' in globals():
                self.video_processor = VideoFrameProcessor(model_path=self.model_path)
                print("Gaze zone analiz modeli yüklendi.")
            else:
                # VideoFrameProcessor yoksa basit bir stub oluştur
                self.video_processor = DummyVideoProcessor()
                print("Gaze zone analiz modeli yerine dummy processor kullanılıyor.")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            self.video_processor = DummyVideoProcessor()
    
    def load_video(self, video_path: str) -> bool:
        """
        Video dosyasını yükle ve temel özelliklerini çıkar.
        
        Args:
            video_path: Video dosyasının yolu
            
        Returns:
            bool: Video başarıyla yüklendiyse True
        """
        if not os.path.exists(video_path):
            print(f"Hata: Video dosyası bulunamadı: {video_path}")
            return False
        
        try:
            # Video dosyasını aç
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Hata: Video dosyası açılamadı: {video_path}")
                return False
            
            # Video özelliklerini al
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            
            # Video bilgilerini yazdır
            print(f"Video yüklendi: {os.path.basename(video_path)}")
            print(f"Boyut: {self.width}x{self.height}, FPS: {self.fps:.2f}")
            print(f"Toplam kare: {self.frame_count}, Süre: {self.duration:.2f} saniye")
            
            # Referansları temizle
            cap.release()
            self.current_video_path = video_path
            
            # Analiz sonuçlarını sıfırla
            self.frame_results = {}
            self.predictions = {}
            
            return True
            
        except Exception as e:
            print(f"Video yüklenirken hata oluştu: {e}")
            return False
    
    def analyze_video(self, sample_rate: int = 1, max_frames: int = 0, 
                     start_time: float = 0, end_time: float = 0) -> Dict:
        """
        Videoyu analiz et ve her kare için gaze zone tahmini yap.
        
        Args:
            sample_rate: Kaç karede bir analiz yapılacağı (1=her kare)
            max_frames: İşlenecek maksimum kare sayısı (0=tümü)
            start_time: Başlangıç zamanı (saniye)
            end_time: Bitiş zamanı (saniye, 0=video sonuna kadar)
            
        Returns:
            Dict: Analiz sonuçları
        """
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            print("Hata: Analiz için yüklenmiş bir video yok.")
            return {}
        
        if not self.video_processor:
            print("Hata: Video işleyici modül hazır değil.")
            return {}
        
        print(f"Video analizi başlatılıyor: {os.path.basename(self.current_video_path)}")
        print(f"Örnekleme oranı: Her {sample_rate} karede bir")
        
        # Video dosyasını aç
        cap = cv2.VideoCapture(self.current_video_path)
        if not cap.isOpened():
            print(f"Hata: Video dosyası açılamadı: {self.current_video_path}")
            return {}
        
        # Başlangıç ve bitiş karelerini hesapla
        start_frame = int(start_time * self.fps) if start_time > 0 else 0
        end_frame = int(end_time * self.fps) if end_time > 0 else self.frame_count
        
        # Max frames kontrolü
        if max_frames > 0:
            end_frame = min(start_frame + max_frames, end_frame)
        
        # Başlangıç karesine git
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Analiz sonuçları
        self.frame_results = {}
        self.predictions = {}
        
        # İlerleme göstergesi için değişkenler
        total_frames = end_frame - start_frame
        processed_count = 0
        start_time = time.time()
        last_progress_update = start_time
        
        # Her kareyi işle
        current_frame = start_frame
        while current_frame < end_frame:
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                break
            
            # Bu kareyi işlememiz gerekiyor mu?
            if (current_frame - start_frame) % sample_rate == 0:
                # Kareyi işle ve gaze zone tahminini yap
                try:
                    # VideoFrameProcessor ile analiz yap
                    results = self.video_processor.process_frame(frame)
                    
                    # Analiz sonuçlarını kaydet
                    frame_id = str(current_frame)
                    self.frame_results[frame_id] = results
                    
                    # Sadece gaze zone tahminini ayır
                    if 'gaze_zone' in results:
                        self.predictions[frame_id] = results['gaze_zone']
                    
                    processed_count += 1
                    
                    # İlerleme durumunu güncelle (saniyede en fazla 1 kez)
                    current_time = time.time()
                    if current_time - last_progress_update >= 1.0:
                        progress = (current_frame - start_frame) / total_frames * 100
                        elapsed = current_time - start_time
                        fps = processed_count / elapsed if elapsed > 0 else 0
                        remaining = (total_frames - (current_frame - start_frame)) / fps if fps > 0 else 0
                        
                        print(f"İlerleme: %{progress:.1f} - "
                              f"İşlenen: {processed_count}/{total_frames//sample_rate} kare - "
                              f"Hız: {fps:.2f} fps - "
                              f"Kalan süre: {remaining:.1f} saniye")
                        
                        last_progress_update = current_time
                
                except Exception as e:
                    print(f"Kare {current_frame} işlenirken hata: {e}")
            
            # Sonraki kareye geç
            current_frame += 1
        
        # Video kaynağını serbest bırak
        cap.release()
        
        # Analiz tamamlandı
        elapsed = time.time() - start_time
        print(f"Video analizi tamamlandı. Toplam {processed_count} kare işlendi.")
        print(f"İşleme süresi: {elapsed:.2f} saniye, Ortalama hız: {processed_count/elapsed:.2f} fps")
        
        return self.predictions
    
    def load_ground_truth(self, json_path: str) -> bool:
        """
        Ground truth verisini yükle.
        
        Args:
            json_path: Ground truth JSON dosyasının yolu
            
        Returns:
            bool: Başarılıysa True
        """
        if not os.path.exists(json_path):
            print(f"Hata: Ground truth dosyası bulunamadı: {json_path}")
            return False
        
        try:
            with open(json_path, 'r') as f:
                self.ground_truth = json.load(f)
            
            print(f"Ground truth verisi yüklendi: {len(self.ground_truth)} kare")
            return True
        
        except Exception as e:
            print(f"Ground truth yüklenirken hata: {e}")
            return False
    
    def save_predictions(self, output_path: str) -> bool:
        """
        Tahminleri JSON formatında kaydet.
        
        Args:
            output_path: Çıktı dosyasının yolu
            
        Returns:
            bool: Başarılıysa True
        """
        if not self.predictions:
            print("Uyarı: Kaydedilecek tahmin sonucu yok.")
            return False
        
        try:
            # Çıktı dizinini oluştur
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # JSON formatında kaydet
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.predictions, f, indent=2, ensure_ascii=False)
            
            print(f"Tahminler kaydedildi: {output_path}")
            return True
        
        except Exception as e:
            print(f"Tahminler kaydedilirken hata: {e}")
            return False
    
    def save_detailed_results(self, output_path: str) -> bool:
        """
        Detaylı analiz sonuçlarını JSON formatında kaydet.
        
        Args:
            output_path: Çıktı dosyasının yolu
            
        Returns:
            bool: Başarılıysa True
        """
        if not self.frame_results:
            print("Uyarı: Kaydedilecek detaylı sonuç yok.")
            return False
        
        try:
            # Çıktı dizinini oluştur
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # JSON formatında kaydet
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.frame_results, f, indent=2, ensure_ascii=False)
            
            print(f"Detaylı sonuçlar kaydedildi: {output_path}")
            return True
        
        except Exception as e:
            print(f"Detaylı sonuçlar kaydedilirken hata: {e}")
            return False
    
    def generate_report(self, output_dir: str) -> Dict:
        """
        Analiz sonuçlarını değerlendir ve bilimsel rapor oluştur.
        
        Args:
            output_dir: Çıktı dizini
            
        Returns:
            Dict: Değerlendirme sonuçları özeti
        """
        if not self.predictions:
            print("Uyarı: Değerlendirme için tahmin sonucu yok.")
            return {}
        
        if not self.ground_truth:
            print("Uyarı: Değerlendirme için ground truth verisi yok.")
            return {}
        
        try:
            # ScientificGazeEvaluator sınıfını kontrol et
            if 'ScientificGazeEvaluator' not in globals():
                print("Hata: Değerlendirme için ScientificGazeEvaluator sınıfı gerekli.")
                return {}
            
            # Değerlendirme sınıfını oluştur
            evaluator = ScientificGazeEvaluator()
            
            # Çıktı dizinini oluştur
            os.makedirs(output_dir, exist_ok=True)
            
            # Değerlendirme yap
            print("Kapsamlı değerlendirme yapılıyor...")
            results = evaluator.comprehensive_analysis(self.ground_truth, self.predictions)
            
            # Akademik rapor oluştur
            print("Akademik rapor oluşturuluyor...")
            report_text = evaluator.generate_academic_report(
                results, "ground_truth.json", "predictions.json"
            )
            
            # Raporu kaydet
            report_file = os.path.join(output_dir, "scientific_evaluation_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # Değerlendirme sonuçlarını JSON olarak kaydet
            results_file = os.path.join(output_dir, "evaluation_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Özet istatistikleri hazırla
            summary = {
                "video_file": os.path.basename(self.current_video_path) if self.current_video_path else "",
                "analyzed_frames": len(self.predictions),
                "ground_truth_frames": len(self.ground_truth),
                "common_frames": len(set(self.predictions.keys()).intersection(set(self.ground_truth.keys()))),
                "overall_accuracy": results.get("basic_metrics", {}).get("overall_accuracy", 0),
                "cohens_kappa": results.get("advanced_metrics", {}).get("cohens_kappa", 0),
                "agreement_strength": results.get("advanced_metrics", {}).get("agreement_strength", "Unknown"),
                "report_file": report_file,
                "results_file": results_file,
            }
            
            print(f"Değerlendirme tamamlandı:")
            print(f"Doğruluk: {summary['overall_accuracy']:.2f}%")
            print(f"Cohen's Kappa: {summary['cohens_kappa']:.4f} ({summary['agreement_strength']})")
            print(f"Rapor dosyası: {report_file}")
            
            return summary
        
        except Exception as e:
            print(f"Rapor oluşturulurken hata: {e}")
            import traceback
            traceback.print_exc()
            return {}


class DummyVideoProcessor:
    """Video işleme modülü mevcut değilse kullanılacak dummy sınıf."""
    
    def __init__(self):
        self.zone_probs = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]  # Örnek olasılıklar
    
    def process_frame(self, frame):
        """Dummy analiz sonuçları döndür."""
        # Rastgele gaze zone tahmini yap (gerçek model yerine)
        zone_probs = np.random.dirichlet(self.zone_probs, 1)[0]
        gaze_zone = np.argmax(zone_probs)
        
        return {
            "gaze_zone": int(gaze_zone),
            "confidence": float(zone_probs[gaze_zone]),
            "zone_probs": zone_probs.tolist(),
            "timestamp": time.time()
        }


# Test ve örnek kullanım
if __name__ == "__main__":
    # Örnek kullanım
    analyzer = VideoGazeAnalyzer()
    
    # Video yükle
    video_path = "sample_video.mp4"
    if os.path.exists(video_path):
        analyzer.load_video(video_path)
        
        # Videoyu analiz et
        analyzer.analyze_video(sample_rate=5, max_frames=1000)
        
        # Sonuçları kaydet
        analyzer.save_predictions("predictions.json")
        
        # Ground truth yükle ve değerlendir
        gt_path = "ground_truth.json"
        if os.path.exists(gt_path):
            analyzer.load_ground_truth(gt_path)
            analyzer.generate_report("evaluation_results") 