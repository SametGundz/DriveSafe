#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sürücü Uykululuk Tespiti Demo Uygulaması

Bu script, ETH-XGaze bakış tahmin modeli ve çeşitli görüntü işleme 
teknikleri kullanarak sürücü uykululuğunu gerçek zamanlı olarak tespit eder.

Kullanım:
    python drowsiness_detection.py [--device cpu|cuda] [--input webcam|video_path]
"""

# Set environment variables to prevent PyQt from creating extra windows
import os
# This ensures we use GTK backend for OpenCV rather than Qt
os.environ["QT_QPA_PLATFORM"] = ""
# Make sure any imported PyQt doesn't create additional windows
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
# Disable MediaPipe visualization
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# Disable TensorFlow debugging/info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

# Debug mode to log window creation attempts
DEBUG_WINDOW_CREATION = False

# Proje kök dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Override window creation functions before any imports
original_imshow = cv2.imshow
def controlled_imshow(winname, mat):
    # Debug logging
    if DEBUG_WINDOW_CREATION:
        print(f"[DEBUG] Window creation attempt: cv2.imshow({winname})")
    
    # Only allow our main window to be created
    if winname == "Sürücü Uykululuk Tespiti":
        return original_imshow(winname, mat)
    # Silently ignore any other windows
    return
cv2.imshow = controlled_imshow

# Override namedWindow to prevent additional windows
original_namedWindow = cv2.namedWindow
def controlled_namedWindow(winname, *args, **kwargs):
    # Debug logging
    if DEBUG_WINDOW_CREATION:
        print(f"[DEBUG] Window creation attempt: cv2.namedWindow({winname})")
        
    if winname == "Sürücü Uykululuk Tespiti":
        return original_namedWindow(winname, *args, **kwargs)
    return
cv2.namedWindow = controlled_namedWindow

# Handle MediaPipe initialization to prevent drawing
import mediapipe as mp
# Override the drawing_utils functions to prevent additional windows
mp_drawing = mp.solutions.drawing_utils
original_draw_landmarks = mp_drawing.draw_landmarks
def controlled_draw_landmarks(*args, **kwargs):
    if DEBUG_WINDOW_CREATION:
        print(f"[DEBUG] MediaPipe draw_landmarks called")
    return None
mp_drawing.draw_landmarks = controlled_draw_landmarks
mp_drawing.draw_detection = lambda *args, **kwargs: None

# Now import our modules
from src.detection.gaze_estimator import GazeEstimator
from src.detection.drowsiness_detector import DrowsinessDetector


def parse_args():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(description="Sürücü Uykululuk Tespiti Demo")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Model inference için cihaz ('cpu' veya 'cuda')")
    parser.add_argument("--input", type=str, default="webcam",
                        help="Giriş kaynağı (video dosyası yolu veya 'webcam')")
    parser.add_argument("--show_gaze", action="store_true", default=True,
                        help="Bakış yönünü görselleştir")
    parser.add_argument("--save_video", action="store_true",
                        help="Çıktı videosunu kaydet")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Çıktı video dosyası (--save_video ile kullanılır)")
    return parser.parse_args()


def draw_gaze(image_in, pitchyaw, origin, length=40.0, thickness=2, color=(0, 255, 255)):
    """
    Bakış yönünü okla görselleştirir.
    
    Args:
        image_in: Giriş resmi
        pitchyaw: [pitch, yaw] açı değerleri
        origin: Ok'un başlangıç noktası
        length: Ok'un uzunluğu
        thickness: Ok'un kalınlığı
        color: Ok'un rengi (BGR)
        
    Returns:
        np.ndarray: Görselleştirilmiş resim
    """
    image_out = image_in.copy()
    
    # Pitch ve yaw açıları kullanarak 3D yön vektörünü 2D'ye projeksiyon yap
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    
    # Ok'un bitiş noktasını hesapla
    end_point = (int(origin[0] + dx), int(origin[1] + dy))
    
    # Ok'u çiz
    cv2.arrowedLine(
        image_out,
        tuple(np.round(origin).astype(np.int32)),
        tuple(np.round(end_point).astype(np.int32)),
        color, thickness, cv2.LINE_AA, tipLength=0.2
    )
    
    return image_out


def load_config():
    """Konfigürasyon dosyasını yükler."""
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Uyarı: Konfigürasyon dosyası yüklenemedi: {config_path}")
        return {}


def get_eye_center_from_landmarks(landmarks, eye_indices):
    """
    Göz işaret noktalarından göz merkezini hesaplar.
    
    Args:
        landmarks: Yüz işaret noktaları listesi
        eye_indices: Göz kontur indeksleri
        
    Returns:
        tuple: Göz merkezi (x, y)
    """
    eye_landmarks = np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in eye_indices])
    eye_center = np.mean(eye_landmarks, axis=0).astype(int)
    return tuple(eye_center)


def main():
    """Ana demo fonksiyonu."""
    # Argümanları ayrıştır
    args = parse_args()
    
    # Konfigürasyonu yükle
    config = load_config()
    
    # MediaPipe Face Mesh modülünü başlat
    print("MediaPipe Face Mesh başlatılıyor...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Gaze Estimator'ı başlat
    print(f"GazeEstimator başlatılıyor...")
    estimator = GazeEstimator()
    
    # Göz kontur noktaları
    # MediaPipe Face Mesh'teki göz konturlarını tanımla
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Drowsiness Detector'ı başlat
    print("DrowsinessDetector başlatılıyor...")
    drowsiness_detector = DrowsinessDetector(config)
    
    # Video girişini ayarla
    if args.input == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.isfile(args.input):
            print(f"Hata: Video dosyası bulunamadı: {args.input}")
            return
        cap = cv2.VideoCapture(args.input)
    
    # Video yakalama başarılı mı kontrol et
    if not cap.isOpened():
        print("Hata: Video girişi açılamadı!")
        return
    
    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Video girişi başarıyla açıldı.")
    print("Çıkmak için 'q' tuşuna basın.")
    
    # Video yazıcıyı ayarla
    video_writer = None
    if args.save_video:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Video kaydı başlatıldı: {args.output}")
    
    # Performans ölçümü için değişkenler
    frame_times = []
    detection_times = []
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    # Ana pencereyi önceden oluştur
    cv2.namedWindow("Sürücü Uykululuk Tespiti", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame_start = time.time()
            
            # Kare yakala
            ret, frame = cap.read()
            if not ret:
                print("Video sonu veya okuma hatası!")
                break
            
            # Görüntüyü işle
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yüz işaretlerini tespit et
            detect_start = time.time()
            results = face_mesh.process(frame_rgb)  # estimator.face_mesh yerine face_mesh kullan
            detect_time = time.time() - detect_start
            detection_times.append(detect_time)
            
            # Görselleştirme için kopyala
            vis_frame = frame.copy()
            
            # Yüz tespit edildi mi?
            if results.multi_face_landmarks:
                try:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    
                    # İşaretleri piksel koordinatlarına dönüştür
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x, y, z = landmark.x * w, landmark.y * h, landmark.z
                        landmarks.append([x, y, z])
                    
                    # Gerekli indekslerin varlığını kontrol et
                    pose_landmarks_indices = [33, 133, 362, 263, 4, 5]
                    if max(pose_landmarks_indices) >= len(landmarks):
                        raise ValueError(f"Gerekli landmark indeksleri bulunamadı. Toplam {len(landmarks)} landmark mevcut.")
                    
                    # Göz kontur indekslerini kontrol et
                    if max(LEFT_EYE_CONTOUR) >= len(landmarks) or max(RIGHT_EYE_CONTOUR) >= len(landmarks):
                        raise ValueError(f"Göz kontur indeksleri bulunamadı. Toplam {len(landmarks)} landmark mevcut.")
                    
                    # Landmarks sözlüğünü GazeEstimator için uygun formata getir
                    gaze_landmarks = {
                        'all_landmarks': landmarks,
                        'left_eye': np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in LEFT_EYE_CONTOUR]),
                        'right_eye': np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in RIGHT_EYE_CONTOUR])
                    }
                    
                    # Baş duruşu tahmini için gerekli 6 özel landmark noktasını seç
                    # Sol göz köşeleri, sağ göz köşeleri, burun noktaları
                    pose_landmarks = np.array([
                        [landmarks[idx][0], landmarks[idx][1]] 
                        for idx in pose_landmarks_indices
                    ])
                    
                    # Baş duruşunu hesapla
                    head_rotation, head_translation = estimator._estimate_head_pose(pose_landmarks)
                    
                    # Bakış yönünü tahmin et
                    success, gaze_angles, face_img = estimator.detect(frame, gaze_landmarks)
                    
                    if success and gaze_angles is not None:
                        # Gaze açılarından 3D vektör oluştur
                        yaw, pitch = gaze_angles
                        yaw_rad = np.radians(yaw)
                        pitch_rad = np.radians(pitch)
                        
                        # 3D gaze vektörünü oluştur
                        avg_gaze = np.array([
                            -np.sin(yaw_rad) * np.cos(pitch_rad),
                            -np.sin(pitch_rad),
                            -np.cos(yaw_rad) * np.cos(pitch_rad)
                        ])
                        avg_gaze = avg_gaze / np.linalg.norm(avg_gaze)  # Normalize et
                    else:
                        avg_gaze = None
                    
                    # Varsayılan olarak sol ve sağ göz bakışlarını avg_gaze olarak ata
                    left_gaze = avg_gaze
                    right_gaze = avg_gaze
                    
                    # EAR değerlerini hesapla
                    left_ear, right_ear = drowsiness_detector.compute_ear_from_landmarks(
                        landmarks, 
                        LEFT_EYE_CONTOUR,
                        RIGHT_EYE_CONTOUR
                    )
                    
                    # Uykululuk durumunu güncelle
                    drowsiness_result = drowsiness_detector.update(
                        ear_left=left_ear,
                        ear_right=right_ear,
                        gaze_vector=avg_gaze,
                        head_rotation=head_rotation
                    )
                    
                    # Göstergeleri çiz
                    # 1. Uykululuk durumu
                    vis_frame = drowsiness_detector.visualize(
                        vis_frame, 
                        ear_left=left_ear, 
                        ear_right=right_ear, 
                        show_metrics=True
                    )
                    
                    # 2. Bakış yönünü çiz (opsiyonel)
                    if args.show_gaze and avg_gaze is not None:
                        # Yüz çerçevesini hesapla
                        face_rect = cv2.boundingRect(
                            np.array([[landmark[0], landmark[1]] for landmark in landmarks]).astype(np.int32)
                        )
                        face_center = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)
                        
                        # Gaz vektöründen pitch ve yaw hesapla
                        pitch = np.arcsin(avg_gaze[1])
                        yaw = np.arctan2(avg_gaze[0], avg_gaze[2])
                        
                        # Bakış yönünü çiz
                        vis_frame = draw_gaze(
                            vis_frame, 
                            np.array([pitch, yaw]), 
                            face_center, 
                            length=100, 
                            color=(0, 255, 255), 
                            thickness=2
                        )
                        
                        # Göz merkezlerini hesapla
                        left_eye_center = get_eye_center_from_landmarks(landmarks, LEFT_EYE_CONTOUR)
                        right_eye_center = get_eye_center_from_landmarks(landmarks, RIGHT_EYE_CONTOUR)
                        
                        # Sol ve sağ göz bakış yönlerini göster
                        if left_gaze is not None:
                            pitch_left = np.arcsin(left_gaze[1])
                            yaw_left = np.arctan2(left_gaze[0], left_gaze[2])
                            vis_frame = draw_gaze(
                                vis_frame, 
                                np.array([pitch_left, yaw_left]), 
                                left_eye_center, 
                                length=50, 
                                color=(255, 0, 0)
                            )
                        
                        if right_gaze is not None:
                            pitch_right = np.arcsin(right_gaze[1])
                            yaw_right = np.arctan2(right_gaze[0], right_gaze[2])
                            vis_frame = draw_gaze(
                                vis_frame, 
                                np.array([pitch_right, yaw_right]), 
                                right_eye_center, 
                                length=50, 
                                color=(0, 0, 255)
                            )
                except Exception as e:
                    # Hata durumunda bilgilendir
                    error_msg = f"Yüz işleme hatası: {str(e)}"
                    print(error_msg)
                    cv2.putText(vis_frame, error_msg, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Yüz bulunamadığında uyarı göster
                cv2.putText(vis_frame, "Yüz tespit edilemedi", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS hesapla
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # FPS göster
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (vis_frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Kare işleme süresini ölç
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Görüntüyü göster (ana pencereye)
            cv2.imshow("Sürücü Uykululuk Tespiti", vis_frame)
            
            # Video kaydı
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            # Kullanıcı girişini kontrol et ('q' tuşu ile çık)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"Hata: {str(e)}")
    finally:
        # Performans istatistiklerini yazdır
        if frame_times:
            print("\nPerformans İstatistikleri:")
            avg_fps = 1.0 / np.mean(frame_times[-100:]) if frame_times else 0
            avg_detect = np.mean(detection_times[-100:]) * 1000 if detection_times else 0
            
            print(f"Ortalama FPS: {avg_fps:.1f}")
            print(f"Ortalama Yüz Tespit Süresi: {avg_detect:.1f} ms")
        
        # Kaynakları serbest bırak
        print("Kaynaklar serbest bırakılıyor...")
        cap.release()
        if video_writer is not None:
            video_writer.release()
        
        # Tüm pencereleri kapat
        print("Pencereler kapatılıyor...")
        cv2.destroyAllWindows()
        # Bazı platformlarda pencereler tam kapanmayabilir, bir kez daha deneyelim
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        print("Demo uygulaması sonlandırıldı.")


if __name__ == "__main__":
    main() 