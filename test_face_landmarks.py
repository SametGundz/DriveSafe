#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yüz işaretlerini (landmarks) tespit eden ve baş duruşunu hesaplayan program.
Bu program kameradan görüntü alarak MediaPipe ile yüz işaretlerini tespit eder,
baş duruşunu (yaw, pitch, roll) hesaplar ve görüntü üzerinde gösterir.
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import time
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

class FaceLandmarksApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Pencere başlığı ve boyutu
        self.setWindowTitle("Yüz İşaretleri ve Baş Duruşu Tespiti")
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
        
        # MediaPipe butonu
        self.mediapipe_button = QPushButton("MediaPipe")
        self.mediapipe_button.setMinimumSize(150, 40)
        self.mediapipe_button.clicked.connect(self.use_mediapipe)
        self.button_layout.addWidget(self.mediapipe_button)
        
        # Baş duruşu gösterme butonu
        self.show_head_pose_button = QPushButton("Baş Duruşunu Göster")
        self.show_head_pose_button.setMinimumSize(180, 40)
        self.show_head_pose_button.setCheckable(True)
        self.show_head_pose_button.setChecked(True)
        self.show_head_pose_button.clicked.connect(self.toggle_head_pose)
        self.button_layout.addWidget(self.show_head_pose_button)
        
        # Durum etiketi
        self.status_label = QLabel("Model: MediaPipe | FPS: 0")
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
        
        # MediaPipe göz ve ağız indeksleri
        # Sol göz işaret noktaları
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        # Sağ göz işaret noktaları
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        # Ağız (iç kontur) işaret noktaları
        self.MOUTH_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # Göz ve ağız indekslerini birleştir
        self.EYES_MOUTH_INDICES = self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES + self.MOUTH_INDICES
        
        # Baş duruşu hesaplama için gerekli noktalar (estimator.py'den alındı)
        self.HEAD_POSE_LANDMARKS = [1, 9, 57, 130, 287, 359]
        
        # 3D model noktaları (estimator.py'den alındı)
        self.model_points = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)
        
        # Kamera matrisi ve distorsiyon katsayıları
        self.focal_length = self.camera_width
        self.center = (self.camera_width / 2, self.camera_height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        
        # FPS hesaplama için değişkenler
        self.prev_time = 0
        self.fps = 0
        
        # Durum değişkenleri
        self.use_mediapipe_model = True  # Başlangıçta MediaPipe kullan
        self.show_head_pose = True  # Baş duruşunu göster
        
        # Aktif buton stilini güncelle
        self.update_button_style()
        
        # Timer oluştur
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # Timer'ı başlatmıyoruz, kamera başlatıldığında başlatılacak
    
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
        
        # Kameranın gerçek boyutlarını al
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Kamera matrisini güncelle
        self.focal_length = actual_width
        self.center = (actual_width / 2, actual_height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype=np.float32)
        
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
    
    def use_mediapipe(self):
        """MediaPipe modelini kullan"""
        self.use_mediapipe_model = True
        self.update_button_style()
    
    def toggle_head_pose(self):
        """Baş duruşu gösterme durumunu değiştir"""
        self.show_head_pose = self.show_head_pose_button.isChecked()
        self.update_button_style()
    
    def update_button_style(self):
        """Aktif butonu vurgula"""
        if self.use_mediapipe_model:
            self.mediapipe_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        else:
            self.mediapipe_button.setStyleSheet("")
            
        if self.show_head_pose:
            self.show_head_pose_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        else:
            self.show_head_pose_button.setStyleSheet("")
    
    def calculate_head_pose(self, landmarks, frame):
        """
        Baş duruşunu hesapla (yaw, pitch, roll) - estimator.py yaklaşımıyla
        
        Args:
            landmarks: MediaPipe yüz işaretleri
            frame: Görüntü karesi
            
        Returns:
            frame: Baş duruşu bilgileri eklenmiş görüntü
            angles: (pitch, yaw, roll) açıları
        """
        h, w, c = frame.shape
        
        # 2D landmark noktalarını al (estimator.py'den alınan indeksler)
        face_coordination_in_image = []
        for idx in self.HEAD_POSE_LANDMARKS:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            face_coordination_in_image.append([x, y])
            # Baş duruşu hesaplama için kullanılan noktaları vurgula
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
        
        # Kamera matrisi güncelle
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distorsiyon katsayıları
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # SolvePnP ile rotasyon vektörünü ve translasyon vektörünü hesapla
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, face_coordination_in_image, cam_matrix, dist_matrix)
        
        if not success:
            return frame, (0, 0, 0)
        
        # Rotasyon matrisini hesapla
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        
        # Euler açılarını hesapla (estimator.py'den alınan fonksiyon)
        angles = self.rotation_matrix_to_angles(rotation_matrix)
        
        if self.show_head_pose:
            # Baş duruşu bilgilerini ekrana yaz (pitch, yaw, roll)
            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), angles)):
                k, v = info
                text = f"{k}: {int(v)}"
                cv2.putText(frame, text, (20, i*30 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
            
            # Baş duruşu eksenleri çiz
            self.draw_axes(frame, rotation_vec, translation_vec)
        
        return frame, angles
    
    def rotation_matrix_to_angles(self, rotation_matrix):
        """
        Rotasyon matrisinden Euler açılarını hesapla (estimator.py'den alındı)
        
        Args:
            rotation_matrix: Rotasyon matrisi
            
        Returns:
            np.array: [pitch, yaw, roll] açıları (derece cinsinden)
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                        rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Derece cinsine çevir
        return np.array([x, y, z]) * 180. / math.pi
    
    def draw_axes(self, frame, rvec, tvec, length=50):
        """
        Baş duruşu eksenlerini çiz
        
        Args:
            frame: Görüntü karesi
            rvec: Rotasyon vektörü
            tvec: Translasyon vektörü
            length: Eksen uzunluğu
            
        Returns:
            frame: Eksenler eklenmiş görüntü
        """
        # Eksen noktalarını oluştur
        axis_points = np.array([
            [0, 0, 0],       # Orijin
            [length, 0, 0],  # X ekseni
            [0, length, 0],  # Y ekseni
            [0, 0, length]   # Z ekseni
        ], dtype=np.float32)
        
        # 3D noktaları 2D'ye projeksiyon yap
        h, w, c = frame.shape
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, cam_matrix, dist_matrix)
        imgpts = imgpts.astype(int)
        
        # Eksenleri çiz
        origin = tuple(imgpts[0].ravel())
        frame = cv2.line(frame, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X ekseni - Kırmızı
        frame = cv2.line(frame, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y ekseni - Yeşil
        frame = cv2.line(frame, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z ekseni - Mavi
        
        return frame
    
    def update_frame(self):
        """Kamera karesini güncelle ve işle"""
        if not self.camera_active or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # FPS hesapla
        curr_time = time.time()
        if self.prev_time > 0:
            self.fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        
        # Ayna etkisi (selfie view)
        frame = cv2.flip(frame, 1)
        
        # Görüntüyü hazırla
        h, w, c = frame.shape
        
        # Durum bilgisini güncelle
        model_name = "MediaPipe"
        self.status_label.setText(f"Model: {model_name} | FPS: {int(self.fps)}")
        
        # MediaPipe ile yüz işaretlerini tespit et
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Baş duruşunu hesapla
                frame, angles = self.calculate_head_pose(face_landmarks.landmark, frame)
                
                # Tüm yüz işaretlerini çiz (nokta olarak)
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # Normalize edilmiş koordinatları piksel koordinatlarına dönüştür
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    # Tüm işaret noktalarını yeşil yap
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # OpenCV BGR -> Qt RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # QImage ve QPixmap oluştur
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Label'a pixmap'i ayarla
        self.camera_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        """Pencere kapatıldığında kaynakları serbest bırak"""
        self.stop_camera()
        self.face_mesh.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FaceLandmarksApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 