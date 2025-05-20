#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Yüz modeli görselleştirme modülü.

Bu modül, baş duruşu (head pose) verilerine göre 3D yüz modelini 
görselleştiren bir widget sağlar.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QCheckBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from collections import deque
import logging
import math
import random
import sys
import time

# Get module-specific logger
logger = logging.getLogger(__name__)

class HeadPoseModelWidget(QOpenGLWidget):
    """
    Baş duruşu verilerine göre 3D yüz modelini görselleştiren widget.
    
    Bu widget, pitch, yaw ve roll açılarına göre 3D modeli döndürür
    ve yüzün mevcut durumunu görselleştirir.
    """
    
    def __init__(self, parent=None, config=None):
        """
        3D yüz modeli widget'ını başlat.
        
        Args:
            parent: Ebeveyn widget
            config: Yapılandırma sözlüğü
        """
        # OpenGL formatını ayarla
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)  # Anti-aliasing
        QSurfaceFormat.setDefaultFormat(fmt)
        
        super().__init__(parent)
        
        self.config = config or {}
        
        # GLUT başlatma durumunu izleme
        self.glut_initialized = False
        
        # Debug görselleştirme ayarı
        self.show_debug_info = False
        
        # Head pose açıları (derece cinsinden)
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        
        # Ham açı değerleri (MediaPipe'den gelen)
        self.raw_pitch = 0.0
        self.raw_yaw = 0.0
        self.raw_roll = 0.0
        
        # Ölçeklendirme faktörü
        self.scale_factor = 1.0
        
        # Hedef açılar (animasyon için)
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.target_roll = 0.0
        
        # Son değerleri izlemek için değişkenler
        self._last_pitch = 0.0
        self._last_yaw = 0.0
        self._last_roll = 0.0
        
        # FPS hesaplama için son kare zamanı
        self._last_frame_time = 0.0
        
        # Görünüm ayarları
        self.x_trans = 0.0
        self.y_trans = 0.0
        self.z_trans = -5.0  # Kameradan uzaklık
        self.scale = 1.0
        
        # Model rengi (varsayılan ten rengi)
        self.model_color = (0.9, 0.8, 0.7)
        
        # Eksen renklerini ayarla (x=kırmızı, y=yeşil, z=mavi)
        self.axes_enabled = True
        self.axes_length = 1.0
        
        # Grid görüntüleme
        self.show_grid = False
        
        # Modelin sınırları
        self.model_bounds = {
            'min_x': -1.0, 'max_x': 1.0,
            'min_y': -1.5, 'max_y': 1.0,
            'min_z': -1.0, 'max_z': 1.0
        }
        
        # Yüz modeli için örnek veri (küre ve silindirler)
        self.face_model = self._generate_face_model()
        
        # Widget'ı boyutlandır
        self.setMinimumSize(200, 200)
        
        # Mouse etkileşimleri için değişkenler
        self.last_pos = None
        self.mouse_pressed = False
        
        # Ölçeklendirme faktörü - baş hareketlerini daha belirgin göstermek için
        self.angle_scale_factor = 1.2
        
        # Açı yumuşatma için geçmiş değerleri tut (deque, sabit boyutlu bir kuyruk sağlar)
        # Gecikmeyi azaltmak için history_size'ı 5'ten 3'e düşür
        self.history_size = 3
        self.pitch_history = deque(maxlen=self.history_size)
        self.yaw_history = deque(maxlen=self.history_size)
        self.roll_history = deque(maxlen=self.history_size)
        
        # Başlangıç değerlerini ekle
        for _ in range(self.history_size):
            self.pitch_history.append(0.0)
            self.yaw_history.append(0.0)
            self.roll_history.append(0.0)
            
        # Performans optimizasyonu: Render durumunu kontrol et
        self.rendering_active = True
        
        # Performans optimizasyonu: Güncelleme zamanını kontrol et
        # Limit updates to 30fps for efficiency
        self.update_timer = QTimer()
        self.update_timer.setInterval(33)  # ~30fps
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start()
        
        # Animasyon timer'ı başlangıçta oluşturma
        self.animation_timer = QTimer()
        self.animation_timer.setInterval(16)  # ~60fps animations
        self.animation_timer.timeout.connect(self._animate_pose)
        
        # OpenGL kaynakları için izleme değişkenleri
        self.gl_initialized = False
        self.display_lists = []
        
        logger.debug("HeadPoseModelWidget initialized")
    
    def _generate_face_model(self):
        """
        Daha ayrıntılı bir 3D yüz modeli oluştur.
        
        Returns:
            dict: 3D yüz modelinin bileşenleri
        """
        # Cilt rengi ve diğer ortak renkler
        skin_color = self.model_color
        eye_white = (1.0, 1.0, 1.0)
        eye_color = (0.3, 0.5, 0.8)  # Mavi gözler
        mouth_color = (0.9, 0.5, 0.5)  # Pembe
        eyebrow_color = (0.2, 0.2, 0.2)  # Koyu kahverengi
        
        # Geliştirilmiş yüz modeli bileşenleri
        face_model = {
            # Ana baş modeli (küre)
            'head': {
                'type': 'sphere',
                'radius': 1.0,
                'position': (0, 0, 0),
                'color': skin_color
            },
            
            # Burun (silindir ve küre)
            'nose_bridge': {
                'type': 'cylinder',
                'radius': 0.15,
                'height': 0.5,
                'position': (0, -0.1, 0.9),
                'rotation': (80, 0, 0),  # x ekseninde 80 derece döndür
                'color': skin_color
            },
            'nose_tip': {
                'type': 'sphere',
                'radius': 0.15,
                'position': (0, -0.3, 1.0),
                'color': skin_color
            },
            
            # Sol göz
            'left_eye': {
                'type': 'sphere',
                'radius': 0.18,
                'position': (-0.3, 0.1, 0.85),
                'color': eye_white
            },
            'left_pupil': {
                'type': 'sphere',
                'radius': 0.08,
                'position': (-0.3, 0.1, 1.03),
                'color': eye_color
            },
            'left_iris': {
                'type': 'sphere',
                'radius': 0.03,
                'position': (-0.3, 0.1, 1.12),
                'color': (0, 0, 0)  # Siyah
            },
            
            # Sağ göz
            'right_eye': {
                'type': 'sphere',
                'radius': 0.18,
                'position': (0.3, 0.1, 0.85),
                'color': eye_white
            },
            'right_pupil': {
                'type': 'sphere',
                'radius': 0.08,
                'position': (0.3, 0.1, 1.03),
                'color': eye_color
            },
            'right_iris': {
                'type': 'sphere',
                'radius': 0.03,
                'position': (0.3, 0.1, 1.12),
                'color': (0, 0, 0)  # Siyah
            },
            
            # Kaşlar
            'left_eyebrow': {
                'type': 'cylinder',
                'radius': 0.05,
                'height': 0.4,
                'position': (-0.3, 0.3, 0.8),
                'rotation': (0, 0, 30),  # z ekseninde 30 derece döndür
                'color': eyebrow_color
            },
            'right_eyebrow': {
                'type': 'cylinder',
                'radius': 0.05,
                'height': 0.4,
                'position': (0.3, 0.3, 0.8),
                'rotation': (0, 0, -30),  # z ekseninde -30 derece döndür
                'color': eyebrow_color
            },
            
            # Ağız
            'mouth': {
                'type': 'cylinder',
                'radius': 0.3,
                'height': 0.1,
                'position': (0, -0.5, 0.7),
                'rotation': (90, 0, 0),  # x ekseninde 90 derece döndür
                'color': mouth_color
            },
            
            # Kulaklar
            'left_ear': {
                'type': 'sphere',
                'radius': 0.2,
                'position': (-1.0, 0, 0),
                'color': skin_color
            },
            'right_ear': {
                'type': 'sphere',
                'radius': 0.2,
                'position': (1.0, 0, 0),
                'color': skin_color
            }
        }
        
        return face_model
    
    def __del__(self):
        """Nesne yok edildiğinde kaynakları temizle"""
        try:
            # OpenGL kaynaklarını serbest bırak
            if self.isValid() and self.gl_initialized:
                self.makeCurrent()
                # Display listleri ve diğer OpenGL kaynaklarını temizle
                for display_list in self.display_lists:
                    if glIsList(display_list):
                        glDeleteLists(display_list, 1)
                self.doneCurrent()
            
            # Timer'ları durdur
            self.release_resources()
                
            logger.debug("HeadPoseModelWidget resources cleaned up in destructor")
        except Exception as e:
            logger.error(f"Error cleaning up OpenGL resources in destructor: {str(e)}")
    
    def closeEvent(self, event):
        """Widget kapatıldığında doğru temizleme işlemleri yap"""
        try:
            # Tüm kaynakları serbest bırak
            self.release_resources()
            logger.debug("HeadPoseModelWidget resources cleaned up in closeEvent")
        except Exception as e:
            logger.error(f"Error in closeEvent: {str(e)}")
        
        # Üst sınıfın closeEvent metodunu çağır
        super().closeEvent(event)
    
    def hideEvent(self, event):
        """Widget gizlendiğinde gereksiz render işlemlerini durdur."""
        self.rendering_active = False
        
        # Animasyon ve güncelleme timer'larını durdur
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        
        # Güncelleme timer'ını durdurma - ama tamamen durdurmak yerine 
        # aralığı uzatarak kaynak kullanımını azalt
        if self.update_timer.isActive():
            self.update_timer.setInterval(500)  # 0.5 saniyede bir güncelle (2 fps)
        
        logger.debug("HeadPoseModelWidget hidden, rendering paused")
        super().hideEvent(event)
    
    def showEvent(self, event):
        """Widget gösterildiğinde render işlemlerini tekrar başlat."""
        self.rendering_active = True
        
        # Timer'ları tekrar orijinal aralıklarına getir
        self.update_timer.setInterval(33)  # ~30fps
        
        logger.debug("HeadPoseModelWidget shown, rendering resumed")
        super().showEvent(event)
    
    def release_resources(self):
        """Tüm kaynakları serbest bırak"""
        # Timer'ları durdur
        if hasattr(self, 'update_timer') and self.update_timer.isActive():
            self.update_timer.stop()
        
        if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
            self.animation_timer.stop()
        
        # Ek kaynakları serbest bırak (ONNX modeli gibi ağır kaynakları)
        # Not: Bu widget doğrudan ONNX kullanmıyor, ancak gerekirse burada
        # diğer ağır kaynaklar temizlenebilir
        
        # Bellek temizliği için garbage collector'ı zorla
        import gc
        gc.collect()
        
        logger.debug("HeadPoseModelWidget resources released")
    
    def initializeGL(self):
        """OpenGL bağlamını başlat ve gerekli ayarları yap."""
        # GL bağlamını başlat
        try:
            # GLUT başlat - güvenli bir şekilde
            try:
                # Boş argüman listesi ile GLUT başlat
                # sys.argv'yi kullanarak gerçek argümanları geçme
                glutInit([])
                self.glut_initialized = True
                logger.debug("GLUT initialized successfully")
            except Exception as glut_error:
                logger.error(f"Error initializing GLUT: {str(glut_error)}")
                self.glut_initialized = False
                
            # Arka plan rengini ayarla (hafif gri tonunda)
            glClearColor(0.95, 0.95, 0.95, 1.0)
            
            # Derinlik testi etkinleştir
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            
            # Işık efektleri etkinleştir
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            
            # Yumuşak gölgeleme etkinleştir (flat shading yerine smooth shading)
            glShadeModel(GL_SMOOTH)
            
            # Işık pozisyonu ve özellikleri
            light_position = [10.0, 10.0, 10.0, 1.0]  # Sağ üst köşeden gelen ışık
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            
            # Ortam ışığı ayarla (hafif mavi tonu)
            ambient_light = [0.2, 0.2, 0.22, 1.0]
            glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
            
            # Yayılan ışık ayarla (beyaz)
            diffuse_light = [0.8, 0.8, 0.8, 1.0]
            glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
            
            # OpenGL başlatıldı olarak işaretle
            self.gl_initialized = True
            
            logger.debug("OpenGL initialized for HeadPoseModelWidget")
        except Exception as e:
            logger.error(f"Error initializing OpenGL: {str(e)}")
    
    def resizeGL(self, width, height):
        """
        OpenGL görünüm boyutlarını ayarla.
        
        Args:
            width: Genişlik
            height: Yükseklik
        """
        if height == 0:
            height = 1
            
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height
        gluPerspective(45, aspect, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def paintGL(self):
        """3D sahneyi çiz."""
        # Eğer rendering pasif ise hiçbir şey çizme
        if not hasattr(self, 'rendering_active') or not self.rendering_active:
            return
            
        # Buffer'ları temizle
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Kamerayı konumlandır
        glTranslatef(self.x_trans, self.y_trans, self.z_trans)
        
        # Head pose açılarına göre modeli döndür - rotasyon sırası önemli!
        # Sıralama: önce yaw, sonra pitch, en son roll (gimbal lock problemini azaltır)
        glRotatef(self.yaw, 0, 1, 0)     # Y-ekseni etrafında dönüş (yaw - sağa/sola dönme)
        glRotatef(self.pitch, 1, 0, 0)   # X-ekseni etrafında dönüş (pitch - yukarı/aşağı bakma)
        glRotatef(self.roll, 0, 0, 1)    # Z-ekseni etrafında dönüş (roll - başı yana yatırma)
        
        # Eksenler (X: kırmızı, Y: yeşil, Z: mavi)
        if self.axes_enabled:
            self._draw_axes()
        
        # Grid göster
        if self.show_grid:
            self._draw_grid()
        
        # Yüz modeli bileşenlerini çiz
        for part_name, part in self.face_model.items():
            if part['type'] == 'sphere':
                self._draw_sphere(part)
            elif part['type'] == 'cylinder':
                self._draw_cylinder(part)
        
        # Head pose bilgilerini göster - daha okunaklı metin düzeni
        # Ekranın sol üst köşesinde göster
        self._draw_text(f"Pitch (X): {self.pitch:.1f}°", -1.5, 1.5, 0)
        self._draw_text(f"Yaw (Y): {self.yaw:.1f}°", -1.5, 1.3, 0)
        self._draw_text(f"Roll (Z): {self.roll:.1f}°", -1.5, 1.1, 0)
        
        # Show debug info if enabled
        if hasattr(self, 'show_debug_info') and self.show_debug_info:
            self._draw_debug_info()
    
    def _draw_axes(self):
        """Koordinat eksenlerini ve etiketlerini çiz."""
        # Işığı devre dışı bırak (daha net görünüm için)
        glDisable(GL_LIGHTING)
        
        # Çizgi kalınlığını ayarla
        glLineWidth(2.0)
        
        # Daha uzun eksenler çiz
        axis_length = self.axes_length * 1.5
        
        glBegin(GL_LINES)
        
        # X ekseni (kırmızı) - Pitch ekseni (yukarı/aşağı)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_length, 0.0, 0.0)
        
        # Y ekseni (yeşil) - Yaw ekseni (sağa/sola)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_length, 0.0)
        
        # Z ekseni (mavi) - Roll ekseni (saat yönü/tersine)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_length)
        
        glEnd()
        
        # Eksen etiketlerini çiz
        glColor3f(1.0, 0.0, 0.0)  # Kırmızı
        self._draw_text("X", axis_length * 1.1, 0, 0)
        
        glColor3f(0.0, 1.0, 0.0)  # Yeşil
        self._draw_text("Y", 0, axis_length * 1.1, 0)
        
        glColor3f(0.0, 0.0, 1.0)  # Mavi
        self._draw_text("Z", 0, 0, axis_length * 1.1)
        
        # Hareket eksenlerini açıklayıcı metin
        glColor3f(0.7, 0.7, 0.7)  # Gri
        self._draw_text("Pitch", axis_length * 0.5, -0.2, 0)
        self._draw_text("Yaw", -0.2, axis_length * 0.5, 0)
        self._draw_text("Roll", -0.2, 0, axis_length * 0.5)
        
        # Varsayılan değerlere geri dön
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_grid(self):
        """3D sahneye grid çiz."""
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        
        # X-Z düzleminde grid çiz
        for i in range(-10, 11):
            glVertex3f(i, -2, -10)
            glVertex3f(i, -2, 10)
            glVertex3f(-10, -2, i)
            glVertex3f(10, -2, i)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _draw_sphere(self, sphere):
        """
        Küre çiz.
        
        Args:
            sphere: Küre özellikleri (radius, position, color)
        """
        x, y, z = sphere['position']
        radius = sphere['radius']
        r, g, b = sphere['color']
        
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(r, g, b)
        glutSolidSphere(radius, 20, 20)
        glPopMatrix()
    
    def _draw_cylinder(self, cylinder):
        """
        Silindir çiz.
        
        Args:
            cylinder: Silindir özellikleri (radius, height, position, rotation, color)
        """
        x, y, z = cylinder['position']
        rx, ry, rz = cylinder.get('rotation', (0, 0, 0))
        radius = cylinder['radius']
        height = cylinder['height']
        r, g, b = cylinder['color']
        
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)
        glRotatef(rz, 0, 0, 1)
        
        glColor3f(r, g, b)
        
        # Silindirin üst ve alt yüzeyleri (diskler)
        glPushMatrix()
        glTranslatef(0, 0, -height/2)
        gluDisk(gluNewQuadric(), 0, radius, 20, 1)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(0, 0, height/2)
        gluDisk(gluNewQuadric(), 0, radius, 20, 1)
        glPopMatrix()
        
        # Silindirin yan yüzeyi
        quad = gluNewQuadric()
        gluCylinder(quad, radius, radius, height, 20, 20)
        
        glPopMatrix()
    
    def _draw_text(self, text, x, y, z):
        """
        Ekrana metin çiz.
        
        Args:
            text: Gösterilecek metin
            x, y, z: Konumu
        """
        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)  # Beyaz renk
        
        glRasterPos3f(x, y, z)
        
        # Sadece GLUT başlatıldıysa metin göster
        if hasattr(self, 'glut_initialized') and self.glut_initialized:
            try:
                for c in text:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
            except Exception as e:
                logger.debug(f"Error rendering text with GLUT: {str(e)}")
        
        glEnable(GL_LIGHTING)
    
    def update_pose(self, pitch, yaw, roll):
        """
        Head pose açılarını güncelle ve yeniden çiz.
        
        Args:
            pitch: X-ekseni etrafında dönüş (derece)
            yaw: Y-ekseni etrafında dönüş (derece)
            roll: Z-ekseni etrafında dönüş (derece)
        """
        try:
            # Ham açı değerlerini sakla (debug gösterimi için)
            self.raw_pitch = pitch
            self.raw_yaw = yaw
            self.raw_roll = roll
            
            # İşaret doğrultmaları - MediaPipe koordinat sisteminden OpenGL koordinat sistemine dönüşüm
            # MediaPipe açılarını OpenGL'e uygun hale getir
            pitch = -pitch  # OpenGL'de pozitif pitch yukarı bakma anlamına gelir
            # yaw işaretini koruyoruz - hem MediaPipe hem OpenGL'de pozitif yaw sola bakmak anlamına gelir
            # roll işaretini koruyoruz - gerekli dönüşüm zaten uygulu
            
            # Değerleri sınırla (aşırı rotasyonları engelle)
            pitch = max(min(pitch, 90), -90)  # -90° ile +90° arasında
            yaw = max(min(yaw, 90), -90)      # -90° ile +90° arasında
            roll = max(min(roll, 45), -45)    # -45° ile +45° arasında (daha sınırlı)
            
            # NaN veya sonsuz değerleri filtrele
            if (math.isnan(pitch) or math.isnan(yaw) or math.isnan(roll) or
                math.isinf(pitch) or math.isinf(yaw) or math.isinf(roll)):
                logger.warning(f"Invalid angle values detected: pitch={pitch}, yaw={yaw}, roll={roll}")
                return
                
            # Gelişmiş dinamik ölçeklendirme - küçük hareketleri daha belirgin, büyük hareketleri daha yumuşak yap
            def dynamic_scale(angle):
                sign = 1 if angle >= 0 else -1
                abs_angle = abs(angle)
                base_scale = 2.0  # Temel ölçeklendirme faktörü
                
                # Scale faktörünü debug gösterimi için sakla
                self.scale_factor = base_scale
                
                if abs_angle < 5.0:
                    # Küçük hareketler için daha hassas
                    return sign * abs_angle * base_scale
                else:
                    # Büyük hareketler için azalan hassasiyet
                    return sign * (5.0 * base_scale + (abs_angle - 5.0) * 0.5)
            
            # Yumuşatma için geçmiş değerlere ekle
            scaled_pitch = dynamic_scale(pitch)
            scaled_yaw = dynamic_scale(yaw)
            scaled_roll = dynamic_scale(roll)
            
            # Önceki değerlerle mevcut değerler arasında büyük bir fark varsa,
            # ani tepki vermek için geçmişi temizle ve sadece yeni değerleri kullan
            # (Gecikmeyi azaltmak için büyük baş hareketlerinde)
            if (self.pitch_history and 
                (abs(scaled_pitch - sum(self.pitch_history) / len(self.pitch_history)) > 10.0 or
                 abs(scaled_yaw - sum(self.yaw_history) / len(self.yaw_history)) > 10.0 or
                 abs(scaled_roll - sum(self.roll_history) / len(self.roll_history)) > 5.0)):
                # Geçmişi temizle
                self.pitch_history.clear()
                self.yaw_history.clear()
                self.roll_history.clear()
                # Yeni değeri 3 kez ekle - ağırlıklı ortalama gibi
                for _ in range(self.history_size):
                    self.pitch_history.append(scaled_pitch)
                    self.yaw_history.append(scaled_yaw)
                    self.roll_history.append(scaled_roll)
                logger.debug("Large head movement detected, immediate response activated")
            else:
                # Normal durumda değerleri geçmişe ekle
                self.pitch_history.append(scaled_pitch)
                self.yaw_history.append(scaled_yaw)
                self.roll_history.append(scaled_roll)
            
            # Yumuşatılmış hedef değerleri hesapla
            smoothed_pitch = sum(self.pitch_history) / len(self.pitch_history)
            smoothed_yaw = sum(self.yaw_history) / len(self.yaw_history)
            smoothed_roll = sum(self.roll_history) / len(self.roll_history)
            
            # Hedef değerleri ayarla
            self.target_pitch = smoothed_pitch
            self.target_yaw = smoothed_yaw
            self.target_roll = smoothed_roll
            
            # Animasyon başlat - eğer çalışmıyorsa
            if not self.animation_timer.isActive():
                self.animation_timer.start()
            
            # Debug logları ekle (düşük detay seviyesi)
            if random.random() < 0.05:  # Logları azaltmak için sadece %5 oranında log tut
                logger.debug(f"Head pose updated - Target Pitch: {self.target_pitch:.1f}, Yaw: {self.target_yaw:.1f}, Roll: {self.target_roll:.1f}")
        except Exception as e:
            logger.error(f"Error updating head pose: {str(e)}")
    
    def _animate_pose(self):
        """Poz geçişlerini yumuşak bir şekilde animasyonla yapar."""
        try:
            # Animasyon hız faktörü - gecikmeyi azaltmak için 0.3'ten 0.5'e yükselt
            speed = 0.5
            
            # Yeni değerleri yumuşak geçişle hesapla
            self.pitch += (self.target_pitch - self.pitch) * speed
            self.yaw += (self.target_yaw - self.yaw) * speed
            self.roll += (self.target_roll - self.roll) * speed
            
            # Son değerleri güncelle
            self._last_pitch = self.pitch
            self._last_yaw = self.yaw
            self._last_roll = self.roll
            
            # Modeli güncelle
            self.update()
            
            # Hedef değere yeterince yakınsa animasyonu durdur
            if (abs(self.target_pitch - self.pitch) < 0.1 and
                abs(self.target_yaw - self.yaw) < 0.1 and
                abs(self.target_roll - self.roll) < 0.1):
                self.animation_timer.stop()
                logger.debug("Head pose animation completed")
        except Exception as e:
            logger.error(f"Error in head pose animation: {str(e)}")
            self.animation_timer.stop()
    
    def mousePressEvent(self, event):
        """Mouse basma olayını işle."""
        self.last_pos = event.position()
        self.mouse_pressed = True
    
    def mouseReleaseEvent(self, event):
        """Mouse bırakma olayını işle."""
        self.mouse_pressed = False
    
    def mouseMoveEvent(self, event):
        """
        Mouse hareket olayını işle.
        
        Kullanıcının mouse ile modeli döndürmesini sağlar.
        """
        if not self.mouse_pressed:
            return
        
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        
        if event.buttons() & Qt.MouseButton.LeftButton:
            # Sol tuş: Yaw ve Pitch değiştir
            self.yaw += dx * 0.5
            self.pitch += dy * 0.5
        elif event.buttons() & Qt.MouseButton.RightButton:
            # Sağ tuş: Roll değiştir
            self.roll += dx * 0.5
        
        self.last_pos = event.position()
        self.update()  # Widget'ı yeniden çiz
    
    def wheelEvent(self, event):
        """
        Mouse tekerleği olayını işle.
        
        Kullanıcının mouse tekerleği ile yakınlaştırıp uzaklaştırmasını sağlar.
        """
        delta = event.angleDelta().y() / 120  # Windows'da standart adım başına 120
        self.z_trans += delta * 0.5
        self.update()  # Widget'ı yeniden çiz
    
    def reset_view(self):
        """Görünümü varsayılan konuma sıfırla."""
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.x_trans = 0.0
        self.y_trans = 0.0
        self.z_trans = -5.0
        
        # Geçmiş değerleri sıfırla
        for _ in range(self.history_size):
            self.pitch_history.append(0.0)
            self.yaw_history.append(0.0)
            self.roll_history.append(0.0)
            
        self.update()  # Widget'ı yeniden çiz
        logger.debug("View reset to default position")

    def _draw_debug_info(self):
        """Draw technical debug information"""
        glDisable(GL_LIGHTING)
        
        # Draw raw values from MediaPipe
        glColor3f(1.0, 0.8, 0.2)  # Amber color for debug info
        self._draw_text(f"Raw Pitch: {self.raw_pitch:.1f}°", -1.5, -1.3, 0)
        self._draw_text(f"Raw Yaw: {self.raw_yaw:.1f}°", -1.5, -1.5, 0)
        self._draw_text(f"Raw Roll: {self.raw_roll:.1f}°", -1.5, -1.7, 0)
        
        # Scale factors for each axis
        base_scale = 2.0  # From dynamic_scale function
        self._draw_text(f"Scale Factor: {base_scale:.1f}x", -1.5, -1.9, 0)
        
        # Draw frame rate
        now = time.time()
        if hasattr(self, '_last_frame_time') and self._last_frame_time > 0:
            fps = 1.0 / (now - self._last_frame_time) if now > self._last_frame_time else 0
            self._draw_text(f"FPS: {fps:.1f}", -1.5, -2.1, 0)
        self._last_frame_time = now
        
        glEnable(GL_LIGHTING)


class Head3DPanel(QWidget):
    """
    3D baş duruşu görselleştirme paneli.
    
    Bu panel, HeadPoseModelWidget'ını içeren bir konteynırdır.
    """
    
    def __init__(self, config, parent=None):
        """
        3D baş modeli panelini başlat.
        
        Args:
            config: Yapılandırma sözlüğü
            parent: Ebeveyn widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Layout oluştur
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Head pose modeli widget'ını oluştur
        self.model_widget = HeadPoseModelWidget(self, config)
        layout.addWidget(self.model_widget)
        
        # Add reset button
        self.reset_button = QPushButton("Reset View", self)
        self.reset_button.clicked.connect(self.reset_view)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        layout.addWidget(self.reset_button)
        
        # Add debug visualization toggle
        self._create_debug_controls()
        
        # Widget görünümünü ayarla
        self.setStyleSheet("""
            background-color: #0a0a0a;
            border: 1px solid #333333;
            border-radius: 4px;
        """)
        
        # Panel boyutunu ayarla - biraz daha yüksek olması gerekiyor (buton ve debug için)
        self.setFixedSize(200, 260)
        
        logger.debug("Head3DPanel initialized")
    
    def _create_debug_controls(self):
        """Create debug controls for developers"""
        self.debug_checkbox = QCheckBox("Show Debug Info", self)
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.toggled.connect(self._toggle_debug_info)
        self.debug_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 10px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
            }
        """)
        self.layout().addWidget(self.debug_checkbox)
    
    def _toggle_debug_info(self, checked):
        """Toggle debug information display"""
        if hasattr(self, 'model_widget'):
            self.model_widget.show_debug_info = checked
            self.model_widget.update()
            logger.debug(f"Debug visualization {'enabled' if checked else 'disabled'}")
    
    def reset_view(self):
        """Reset the model to default orientation"""
        if hasattr(self, 'model_widget'):
            self.model_widget.pitch = 0.0
            self.model_widget.yaw = 0.0
            self.model_widget.roll = 0.0
            self.model_widget.x_trans = 0.0
            self.model_widget.y_trans = 0.0
            self.model_widget.z_trans = -5.0
            
            # Clear history and reset target angles
            self.model_widget.pitch_history.clear()
            self.model_widget.yaw_history.clear()
            self.model_widget.roll_history.clear()
            
            # Re-initialize history with zeros
            for _ in range(self.model_widget.history_size):
                self.model_widget.pitch_history.append(0.0)
                self.model_widget.yaw_history.append(0.0)
                self.model_widget.roll_history.append(0.0)
                
            # Reset target angles
            self.model_widget.target_pitch = 0.0
            self.model_widget.target_yaw = 0.0
            self.model_widget.target_roll = 0.0
            
            self.model_widget.update()
            logger.debug("3D model view reset")
            
    def update_pose(self, pitch, yaw, roll):
        """
        Head pose açılarını güncelle.
        
        Args:
            pitch: X-ekseni etrafında dönüş (derece)
            yaw: Y-ekseni etrafında dönüş (derece)
            roll: Z-ekseni etrafında dönüş (derece)
        """
        self.model_widget.update_pose(pitch, yaw, roll)