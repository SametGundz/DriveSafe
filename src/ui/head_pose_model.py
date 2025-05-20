#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Yüz modeli görselleştirme modülü.

Bu modül, baş duruşu (head pose) verilerine göre 3D yüz modelini 
görselleştiren bir widget sağlar.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from collections import deque
import logging
import math
import random

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
        
        # Head pose açıları (derece cinsinden)
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        
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
        self.history_size = 5
        self.pitch_history = deque(maxlen=self.history_size)
        self.yaw_history = deque(maxlen=self.history_size)
        self.roll_history = deque(maxlen=self.history_size)
        
        # Başlangıç değerlerini ekle
        for _ in range(self.history_size):
            self.pitch_history.append(0.0)
            self.yaw_history.append(0.0)
            self.roll_history.append(0.0)
        
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
    
    def initializeGL(self):
        """OpenGL için başlangıç ayarlarını yap."""
        try:
            glutInit()
            glClearColor(0.0, 0.0, 0.1, 1.0)  # Koyu mavi arka plan
            glClearDepth(1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_COLOR_MATERIAL)
            glShadeModel(GL_SMOOTH)
            
            # Işık ayarları
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 10, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1])
            
            logger.debug("OpenGL initialization successful")
        except Exception as e:
            logger.error(f"OpenGL initialization error: {str(e)}")
    
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
        
        # Hareket yönlerini göster - kullanıcıya yol gösterici bilgiler
        if abs(self.pitch) > 15:
            direction = "aşağı" if self.pitch > 0 else "yukarı"
            self._draw_text(f"Baş {direction} dönük", -1.5, -1.3, 0)
        
        if abs(self.yaw) > 15:
            direction = "sola" if self.yaw > 0 else "sağa"
            self._draw_text(f"Baş {direction} dönük", -1.5, -1.5, 0)
    
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
        
        for c in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
        
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
            # İşaret doğrultmaları - MediaPipe koordinat sisteminden OpenGL koordinat sistemine dönüşüm
            # Not: MediaPipe'dan gelen açılar ve OpenGL rotasyon sistemi farklı eksen tanımları kullanır
            
            # MediaPipe açıları: 
            # - pitch: yukarı/aşağı bakma (pozitif aşağı)
            # - yaw: sola/sağa bakma (pozitif sola)
            # - roll: başı yana yatırma (pozitif sağa)
            
            # OpenGL açıları (sağ el kuralına göre):
            # - X ekseni rotasyonu (pitch): pozitif yukarı bakar
            # - Y ekseni rotasyonu (yaw): pozitif sola bakar
            # - Z ekseni rotasyonu (roll): pozitif saat yönünün tersine

            # İşaret düzeltmeleri
            pitch = -pitch  # OpenGL'de pozitif pitch yukarı bakma anlamına gelir
            # yaw işaretini koruyoruz - hem MediaPipe hem OpenGL'de pozitif yaw sola bakmak anlamına gelir
            # roll işaretini koruyoruz - gerekli dönüşüm zaten uygulu
            
            # Değerleri sınırla (aşırı rotasyonları engelle)
            pitch = max(min(pitch, 90), -90)  # -90° ile +90° arasında
            yaw = max(min(yaw, 90), -90)      # -90° ile +90° arasında 
            roll = max(min(roll, 45), -45)    # -45° ile +45° arasında (daha sınırlı)
            
            # Açıları ölçeklendir - kullanıcı deneyimini artırmak için
            # Ölçeklendirme açıların daha belirgin olmasını sağlar
            scaled_pitch = pitch * self.angle_scale_factor
            scaled_yaw = yaw * self.angle_scale_factor
            scaled_roll = roll * self.angle_scale_factor
            
            # NaN veya sonsuz değerleri filtrele
            if (math.isnan(scaled_pitch) or math.isnan(scaled_yaw) or math.isnan(scaled_roll) or
                math.isinf(scaled_pitch) or math.isinf(scaled_yaw) or math.isinf(scaled_roll)):
                logger.warning(f"Invalid angle values detected: pitch={pitch}, yaw={yaw}, roll={roll}")
                return
            
            # Yumuşatma için geçmiş değerlere ekle
            self.pitch_history.append(scaled_pitch)
            self.yaw_history.append(scaled_yaw)
            self.roll_history.append(scaled_roll)
            
            # Yumuşatılmış değerleri hesapla (hareketli ortalama)
            self.pitch = sum(self.pitch_history) / len(self.pitch_history)
            self.yaw = sum(self.yaw_history) / len(self.yaw_history)
            self.roll = sum(self.roll_history) / len(self.roll_history)
            
            # Widget'ı yeniden çiz
            self.update()
            
            # Debug logları ekle (düşük detay seviyesi)
            if random.random() < 0.05:  # Logları azaltmak için sadece %5 oranında log tut
                logger.debug(f"Head pose updated - Pitch: {self.pitch:.1f}, Yaw: {self.yaw:.1f}, Roll: {self.roll:.1f}")
        except Exception as e:
            logger.error(f"Error updating head pose: {str(e)}")
    
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
        
        # Widget görünümünü ayarla
        self.setStyleSheet("""
            background-color: #0a0a0a;
            border: 1px solid #333333;
            border-radius: 4px;
        """)
        
        # Panel boyutunu ayarla
        self.setFixedSize(200, 200)
        
        logger.debug("Head3DPanel initialized")
    
    def update_pose(self, pitch, yaw, roll):
        """
        Head pose açılarını güncelle.
        
        Args:
            pitch: X-ekseni etrafında dönüş (derece)
            yaw: Y-ekseni etrafında dönüş (derece)
            roll: Z-ekseni etrafında dönüş (derece)
        """
        self.model_widget.update_pose(pitch, yaw, roll)