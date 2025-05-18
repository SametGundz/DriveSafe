#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH-XGaze Bakış Yönü Tahmini Test Programı

Bu program, ETH-XGaze modelini kullanarak kameradan alınan görüntülerle
gerçek zamanlı bakış yönü tahmini yapar.

ONNX modelini kullanarak daha hızlı çıkarım sağlar.
"""

import os
import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
import time
import collections
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

class GazeEstimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Pencere başlığı ve boyutu
        self.setWindowTitle("ETH-XGaze Bakış Yönü Tahmini")
        self.setMinimumSize(800, 600)
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Kamera görüntüsü için label
        self.camera_label = QLabel("Kamera başlatılmadı")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.main_layout.addWidget(self.camera_label)
        
        # Butonlar için layout
        self.button_layout = QHBoxLayout()
        
        # Kamera başlat/durdur butonu
        self.camera_button = QPushButton("Kamerayı Başlat")
        self.camera_button.setMinimumSize(150, 40)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.button_layout.addWidget(self.camera_button)
        
        # Bakış yönü gösterme butonu
        self.show_gaze_button = QPushButton("Bakış Yönünü Göster")
        self.show_gaze_button.setMinimumSize(180, 40)
        self.show_gaze_button.setCheckable(True)
        self.show_gaze_button.setChecked(True)
        self.show_gaze_button.clicked.connect(self.toggle_gaze)
        self.button_layout.addWidget(self.show_gaze_button)
        
        # Durum etiketi
        self.status_label = QLabel("Model: ETH-XGaze | FPS: 0")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_layout.addWidget(self.status_label)
        
        # Layout'u ana layout'a ekle
        self.main_layout.addLayout(self.button_layout)
        
        # MediaPipe Face Mesh başlat
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Kamera değişkenleri
        self.cap = None
        self.camera_active = False
        self.camera_id = 0
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 30
        
        # ONNX model yükleme
        self.model_path = os.path.join('models', 'eth_xgaze_model.onnx')
        if os.path.exists(self.model_path):
            self.onnx_session = onnxruntime.InferenceSession(self.model_path)
            print(f"ONNX model yüklendi: {self.model_path}")
        else:
            print(f"HATA: ONNX model bulunamadı: {self.model_path}")
            self.onnx_session = None
        
        # Yüz modeli (3D landmark noktaları)
        # ETH-XGaze'den alınan standart yüz modeli
        self.face_model = np.array([
            [-34.04, 39.55, 57.49],  # Sağ göz dış köşesi
            [-16.1, 42.4, 67.53],    # Sağ göz iç köşesi
            [16.1, 42.4, 67.53],     # Sol göz iç köşesi
            [34.04, 39.55, 57.49],   # Sol göz dış köşesi
            [0.0, 0.0, 8.0],         # Burun ucu
            [0.0, -48.0, 21.0],      # Çene ucu
        ])
        
        # Bakış yönü gösterme durumu
        self.show_gaze = True
        
        # FPS hesaplama için değişkenler
        self.prev_time = 0
        self.fps = 0
        self.fps_history = collections.deque(maxlen=100)  # Son 100 FPS değerini sakla
        self.model_inference_times = collections.deque(maxlen=100)  # Model çıkarım sürelerini sakla
        self.total_frames = 0
        self.total_time = 0
        self.min_fps = float('inf')
        self.max_fps = 0
        
        # FPS analiz sonuçlarını göstermek için label
        self.fps_analysis_label = QLabel("FPS Analizi: Henüz veri yok")
        self.fps_analysis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.fps_analysis_label)
        
        # Timer oluştur
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def toggle_camera(self):
        """Kamerayı başlat veya durdur"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Kamerayı başlat"""
        if self.camera_active:
            return
            
        # Kamerayı başlat
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.camera_label.setText("Kamera açılamadı!")
            return
            
        # Kamera ayarları
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        
        # Timer'ı başlat
        self.timer.start(33)  # ~30 FPS
        self.camera_active = True
        
        # Buton metnini güncelle
        self.camera_button.setText("Kamerayı Durdur")
        self.camera_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
    
    def stop_camera(self):
        """Kamerayı durdur"""
        if not self.camera_active:
            return
            
        # Timer'ı durdur
        self.timer.stop()
        
        # Kamerayı serbest bırak
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Kamera durumunu güncelle
        self.camera_active = False
        
        # Kamera etiketini sıfırla
        self.camera_label.setText("Kamera durduruldu")
        
        # Buton metnini güncelle
        self.camera_button.setText("Kamerayı Başlat")
        self.camera_button.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;")
    
    def toggle_gaze(self):
        """Bakış yönü gösterme durumunu değiştir"""
        self.show_gaze = self.show_gaze_button.isChecked()
        
        # Buton stilini güncelle
        if self.show_gaze:
            self.show_gaze_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        else:
            self.show_gaze_button.setStyleSheet("")
    
    def estimate_head_pose(self, landmarks, face_model, camera_matrix, distortion):
        """
        Baş duruşunu tahmin et
        
        Args:
            landmarks: 2D yüz işaret noktaları
            face_model: 3D yüz modeli
            camera_matrix: Kamera matrisi
            distortion: Distorsiyon katsayıları
            
        Returns:
            rvec, tvec: Rotasyon ve translasyon vektörleri
        """
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera_matrix, distortion, flags=cv2.SOLVEPNP_EPNP)
        
        # Daha fazla optimize et
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera_matrix, distortion, rvec, tvec, True)
        
        return rvec, tvec
    
    def normalize_face(self, img, face_model, landmarks, hr, ht, camera_matrix):
        """
        Yüz görüntüsünü normalize et (ETH-XGaze'den uyarlandı)
        
        Args:
            img: Giriş görüntüsü
            face_model: 3D yüz modeli
            landmarks: 2D yüz işaret noktaları
            hr: Baş duruşu rotasyon vektörü
            ht: Baş duruşu translasyon vektörü
            camera_matrix: Kamera matrisi
            
        Returns:
            img_normalized: Normalize edilmiş yüz görüntüsü
        """
        # Normalize edilmiş kamera parametreleri
        focal_norm = 960  # Normalize edilmiş kameranın odak uzaklığı
        distance_norm = 600  # Göz ve kamera arasındaki normalize edilmiş mesafe
        roi_size = (224, 224)  # Kırpılmış göz görüntüsünün boyutu
        
        # İşaret noktalarının 3D pozisyonlarını hesapla
        ht = ht.reshape((3, 1))
        hR = cv2.Rodrigues(hr)[0]  # Rotasyon matrisi
        Fc = np.dot(hR, face_model.T) + ht  # Yüz modelini döndür ve taşı
        
        # Yüz merkezini bul (göz ve burun merkezlerinin ortalaması)
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
        
        # Görüntüyü normalize et
        distance = np.linalg.norm(face_center)  # Göz ve orijinal kamera arasındaki gerçek mesafe
        
        z_scale = distance_norm / distance
        cam_norm = np.array([  # Sanal kameranın iç parametreleri
            [focal_norm, 0, roi_size[0] / 2],
            [0, focal_norm, roi_size[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([  # Ölçekleme matrisi
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # Rotasyon matrisi R
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))  # Dönüşüm matrisi
        
        img_warped = cv2.warpPerspective(img, W, roi_size)  # Giriş görüntüsünü warp et
        
        return img_warped
    
    def preprocess_image(self, image):
        """
        Görüntüyü model için ön işleme tabi tut
        
        Args:
            image: BGR formatında giriş görüntüsü
            
        Returns:
            preprocessed_img: Ön işleme tabi tutulmuş görüntü
        """
        # BGR'den RGB'ye dönüştür
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü yeniden boyutlandır
        image = cv2.resize(image, (224, 224))
        
        # Normalize et (ImageNet mean ve std değerleri)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        image = (image - mean) / std
        
        # NCHW formatına dönüştür (batch_size, channels, height, width)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        return image
    
    def draw_gaze(self, image, pitchyaw, origin, length=50, thickness=2, color=(0, 0, 255)):
        """
        Bakış yönünü görüntü üzerinde çiz
        
        Args:
            image: Giriş görüntüsü
            pitchyaw: (pitch, yaw) açıları (radyan)
            origin: Bakış vektörünün başlangıç noktası (x, y)
            length: Bakış vektörünün uzunluğu
            thickness: Çizgi kalınlığı
            color: Çizgi rengi (BGR)
            
        Returns:
            image: Bakış vektörü çizilmiş görüntü
        """
        pitch, yaw = pitchyaw
        
        # Küresel koordinatları kartezyen koordinatlara dönüştür
        dx = -length * np.sin(yaw) * np.cos(pitch)
        dy = -length * np.sin(pitch)
        
        # Vektörün bitiş noktasını hesapla
        end_point = (int(origin[0] + dx), int(origin[1] + dy))
        
        # Oku çiz
        cv2.arrowedLine(image, origin, end_point, color, thickness, cv2.LINE_AA, tipLength=0.2)
        
        return image
    
    def update_frame(self):
        """Kamera karesini güncelle ve işle"""
        if not self.camera_active or self.cap is None:
            return
        
        # Frame işleme başlangıç zamanını kaydet
        frame_start_time = time.time()
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # FPS hesapla
        curr_time = time.time()
        if self.prev_time > 0:
            time_diff = curr_time - self.prev_time
            self.fps = 1.0 / time_diff
            self.fps_history.append(self.fps)
            self.total_frames += 1
            self.total_time += time_diff
            
            # Min ve max FPS değerlerini güncelle
            self.min_fps = min(self.min_fps, self.fps)
            self.max_fps = max(self.max_fps, self.fps)
            
            # FPS analiz sonuçlarını güncelle
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            avg_inference_time = sum(self.model_inference_times) / len(self.model_inference_times) if self.model_inference_times else 0
            
            fps_analysis = f"FPS: {self.fps:.1f} | Ort: {avg_fps:.1f} | Min: {self.min_fps:.1f} | Max: {self.max_fps:.1f} | Model Süresi: {avg_inference_time*1000:.1f} ms"
            self.fps_analysis_label.setText(fps_analysis)
            
        self.prev_time = curr_time
        
        # Ayna etkisi (selfie view)
        frame = cv2.flip(frame, 1)
        
        # Görüntüyü hazırla
        h, w, c = frame.shape
        
        # Durum bilgisini güncelle
        self.status_label.setText(f"Model: ETH-XGaze | FPS: {int(self.fps)}")
        
        # MediaPipe ile yüz işaretlerini tespit et
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks and self.onnx_session is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Yüz işaretlerini çıkar
                landmarks = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))
                
                # Göz köşeleri ve burun köşeleri için işaret noktalarını seç
                # MediaPipe indeksleri: Sağ göz dış köşe, Sağ göz iç köşe, Sol göz iç köşe, Sol göz dış köşe, Burun ucu, Çene
                landmark_indices = [33, 133, 362, 263, 1, 199]
                selected_landmarks = []
                
                for idx in landmark_indices:
                    selected_landmarks.append(landmarks[idx])
                
                selected_landmarks = np.array(selected_landmarks, dtype=np.float32)
                
                # Kamera matrisi oluştur
                focal_length = 1.0 * w
                camera_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Distorsiyon katsayıları
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                
                # Baş duruşunu tahmin et
                face_model_pts = self.face_model.reshape(6, 1, 3)
                landmarks_pts = selected_landmarks.reshape(6, 1, 2)
                
                try:
                    rvec, tvec = self.estimate_head_pose(landmarks_pts, face_model_pts, camera_matrix, dist_coeffs)
                    
                    # Yüzü normalize et
                    normalized_face = self.normalize_face(frame, self.face_model, landmarks_pts, rvec, tvec, camera_matrix)
                    
                    # Normalize edilmiş görüntüyü ön işleme tabi tut
                    preprocessed_face = self.preprocess_image(normalized_face)
                    
                    # Model çıkarım süresini ölç
                    model_start_time = time.time()
                    
                    # ONNX modeli ile bakış yönünü tahmin et
                    input_name = self.onnx_session.get_inputs()[0].name
                    output_name = self.onnx_session.get_outputs()[0].name
                    gaze_prediction = self.onnx_session.run([output_name], {input_name: preprocessed_face})[0][0]
                    
                    # Model çıkarım süresini kaydet
                    model_inference_time = time.time() - model_start_time
                    self.model_inference_times.append(model_inference_time)
                    
                    # Pitch ve yaw açılarını al
                    pitch, yaw = gaze_prediction
                    
                    # Bakış yönünü görüntü üzerine çiz
                    if self.show_gaze:
                        # Normalize edilmiş yüz üzerinde bakış yönünü çiz
                        normalized_face_with_gaze = self.draw_gaze(
                            normalized_face.copy(), 
                            (pitch, yaw), 
                            (normalized_face.shape[1] // 2, normalized_face.shape[0] // 2)
                        )
                        
                        # Normalize edilmiş yüzü ana görüntüye yerleştir
                        norm_h, norm_w = normalized_face_with_gaze.shape[:2]
                        frame[0:norm_h, 0:norm_w] = normalized_face_with_gaze
                        
                        # Orijinal görüntü üzerinde bakış yönünü çiz
                        nose_point = (landmarks[1][0], landmarks[1][1])  # Burun ucu
                        frame = self.draw_gaze(frame, (pitch, yaw), nose_point, length=100)
                        
                        # Açıları görüntü üzerine yaz
                        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Model çıkarım süresini görüntü üzerine yaz
                        cv2.putText(frame, f"Model: {model_inference_time*1000:.1f} ms", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Hata: {e}")
        
        # OpenCV BGR -> Qt RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # QImage ve QPixmap oluştur
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Label'a pixmap'i ayarla
        self.camera_label.setPixmap(pixmap)
        
        # Toplam frame işleme süresini hesapla
        frame_processing_time = time.time() - frame_start_time
    
    def closeEvent(self, event):
        """Pencere kapatıldığında kaynakları serbest bırak ve performans raporunu göster"""
        self.stop_camera()
        self.face_mesh.close()
        
        # Performans raporu
        if self.total_frames > 0:
            avg_fps = self.total_frames / self.total_time if self.total_time > 0 else 0
            avg_inference_time = sum(self.model_inference_times) / len(self.model_inference_times) if self.model_inference_times else 0
            
            print("\n===== PERFORMANS RAPORU =====")
            print(f"Toplam işlenen kare sayısı: {self.total_frames}")
            print(f"Toplam geçen süre: {self.total_time:.2f} saniye")
            print(f"Ortalama FPS: {avg_fps:.2f}")
            print(f"Minimum FPS: {self.min_fps:.2f}")
            print(f"Maksimum FPS: {self.max_fps:.2f}")
            print(f"Ortalama model çıkarım süresi: {avg_inference_time*1000:.2f} ms")
            print("============================\n")
        
        event.accept()

def main():
    app = QApplication([])
    window = GazeEstimationApp()
    window.show()
    app.exec()

if __name__ == "__main__":
    main() 