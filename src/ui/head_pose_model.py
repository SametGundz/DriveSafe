#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Yüz modeli görselleştirme modülü.

Bu modül, baş duruşu (head pose) verilerine göre 3D yüz modelini 
görselleştiren bir widget sağlar.
"""

import numpy as np
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QCheckBox, QSlider, QLabel, QHBoxLayout, QSizePolicy
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
import pywavefront
from PIL import Image

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
        
        # Görünüm ayarları - optimizasyon için değerler güncellendi
        self.x_trans = 0.0
        self.y_trans = 0.0
        self.z_trans = -5.0  # Daha yakın kamera konumu (daha önce -8.0 idi)
        self.scale = 1.0
        
        # Modelin yönü (varsayılan olarak normal pozisyon)
        self.model_flip = False  # 180 derece çevirme devre dışı bırakıldı
        self.flip_angle = 180.0
        
        # Model rengi (varsayılan ten rengi)
        self.model_color = (0.9, 0.8, 0.7)
        
        # Eksen renklerini ayarla (x=kırmızı, y=yeşil, z=mavi)
        self.axes_enabled = True
        self.axes_length = 1.0
        
        # Grid görüntüleme
        self.show_grid = False
        
        # OBJ Model ile ilgili değişkenler
        self.face_obj = None
        self.face_texture_id = None
        self.face_display_list = None
        self.use_obj_model = True  # OBJ model kullanımını kontrol eden bayrak
        
        # OBJ model yolları
        self.face_obj_path = os.path.join("models", "face", "face.obj")
        self.face_texture_path = os.path.join("models", "face", "face_texture.png")
        
        # OBJ model yükleme
        try:
            self.face_obj = pywavefront.Wavefront(
                self.face_obj_path,
                create_materials=True,
                collect_faces=True,
                parse=True,
                strict=False  # Daha esnek OBJ ayrıştırma
            )
            logger.info(f"3D face model loaded: {self.face_obj_path} with {len(self.face_obj.meshes)} meshes")
            
            # Model ölçeği ve konumunu ayarla
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            # Modelin sınırlarını bul - vertices'i doğrudan kullan
            for vertex in self.face_obj.vertices:
                if len(vertex) >= 3:
                    x, y, z = vertex
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    min_z = min(min_z, z)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    max_z = max(max_z, z)
            
            # Modelin merkezini ve ölçeğini ayarla
            if min_x != float('inf'):
                self.model_center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
                self.model_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
                logger.info(f"Model center: {self.model_center}, size: {self.model_size}")
                
                # Modelin viewport içindeki konumunu optimize et
                self._optimize_model_position()
            else:
                self.model_center = (0, 0, 0)
                self.model_size = 2.0
                
        except Exception as e:
            self.use_obj_model = False
            logger.error(f"Error loading 3D face model: {str(e)}")
            logger.warning("Falling back to simple face model")
        
        # Modelin sınırları
        self.model_bounds = {
            'min_x': -1.0, 'max_x': 1.0,
            'min_y': -1.5, 'max_y': 1.0,
            'min_z': -1.0, 'max_z': 1.0
        }
        
        # Yüz modeli için örnek veri (küre ve silindirler) - fallback için
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
    
    def _optimize_model_position(self):
        """
        Modelin viewport içindeki konumunu otomatik olarak optimize eder.
        Bu, model boyutuna ve widget büyüklüğüne göre en uygun konumlandırma yapar.
        """
        try:
            # Model boyutuna göre z pozisyonunu optimize et
            # Daha büyük modeller için daha uzak z pozisyonu
            if hasattr(self, 'model_size') and self.model_size > 0:
                # Model büyüklüğüne göre optimal kamera mesafesi
                optimal_distance = self.model_size * 2.5
                self.z_trans = -optimal_distance
                
                # Y pozisyonunu modelin merkezine göre ayarla
                if hasattr(self, 'model_center'):
                    self.y_trans = -self.model_center[1] * 0.5
                    
                logger.debug(f"Model position optimized: z={self.z_trans}, y={self.y_trans}")
        except Exception as e:
            logger.error(f"Error optimizing model position: {str(e)}")
            # Varsayılan değerlere dön
            self.z_trans = -5.0
            self.y_trans = 0.0
    
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
                
                # Doku kaynaklarını temizle
                if hasattr(self, 'face_texture_id') and self.face_texture_id:
                    glDeleteTextures([self.face_texture_id])
                
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
        
        # OBJ modeli kaynakları serbest bırak
        self.face_obj = None
        
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
                
            # Arka plan rengini ayarla (daha koyu bir ton)
            glClearColor(0.1, 0.1, 0.1, 1.0)
            
            # Derinlik testi etkinleştir
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            
            # Işık efektleri etkinleştir
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            
            # Yumuşak gölgeleme etkinleştir (flat shading yerine smooth shading)
            glShadeModel(GL_SMOOTH)
            
            # Işık pozisyonu ve özellikleri - daha yumuşak ışıklar için güncellenmiş değerler
            light_position = [5.0, 5.0, 5.0, 1.0]  # Yukarıdan ve önden gelen ışık
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            
            # Ortam ışığı ayarla (daha düşük ton)
            ambient_light = [0.2, 0.2, 0.2, 1.0]
            glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
            
            # Yayılan ışık ayarla (daha yumuşak beyaz)
            diffuse_light = [0.6, 0.6, 0.6, 1.0]
            glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
            
            # Speküler ışık (modelin parlak noktalarını azaltmak için)
            specular_light = [0.3, 0.3, 0.3, 1.0]
            glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
            
            # İkinci bir ışık kaynağı (daha yumuşak dolgu ışığı)
            glEnable(GL_LIGHT1)
            light1_position = [-5.0, -2.0, 5.0, 1.0]  # Aşağıdan ve arkadan gelen ışık
            glLightfv(GL_LIGHT1, GL_POSITION, light1_position)
            glLightfv(GL_LIGHT1, GL_AMBIENT, [0.05, 0.05, 0.05, 1.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])
            
            # OBJ modeli için doku ve display list oluştur
            if self.use_obj_model and self.face_obj:
                # Doku yükle
                self.face_texture_id = self._load_texture()
                
                # OBJ modeli için display list oluştur
                self.face_display_list = self._create_obj_display_list()
                
                if self.face_display_list:
                    # Display list'i kaynaklara ekle
                    self.display_lists.append(self.face_display_list)
                    logger.info("OBJ model display list created successfully")
                else:
                    # Geri dönüş mekanizması için basit modele geç
                    self.use_obj_model = False
                    logger.warning("Failed to create OBJ model display list, using simple model")
            
            # OpenGL başlatıldı olarak işaretle
            self.gl_initialized = True
            
            logger.debug("OpenGL initialized for HeadPoseModelWidget")
        except Exception as e:
            logger.error(f"Error initializing OpenGL: {str(e)}")
            # OBJ model kullanımını devre dışı bırak
            self.use_obj_model = False
    
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
            
        try:
            # Buffer'ları temizle
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            
            # Kamerayı konumlandır
            glTranslatef(self.x_trans, self.y_trans, self.z_trans)
            
            # Eğer modeli çevirmemiz gerekiyorsa (gerçek dünya hareketlerini yansıtmak için)
            if hasattr(self, 'model_flip') and self.model_flip:
                # Modeli 180 derece döndür - gerçek baş hareketlerini doğru yansıtmak için
                glRotatef(self.flip_angle, 0, 1, 0)  # Y ekseninde 180 derece döndür
            
            # Head pose açılarına göre modeli döndür - rotasyon sırası önemli!
            # Sıralama: önce yaw, sonra pitch, en son roll
            glRotatef(self.yaw, 0, 1, 0)      # Y-ekseni etrafında dönüş (yaw - sağa/sola dönme)
            glRotatef(self.pitch, 1, 0, 0)    # X-ekseni etrafında dönüş (pitch - yukarı/aşağı bakma)
            glRotatef(self.roll, 0, 0, 1)     # Z-ekseni etrafında dönüş (roll - başı yana yatırma)
            
            # Eksenler (X: kırmızı, Y: yeşil, Z: mavi)
            if self.axes_enabled:
                self._draw_axes()
            
            # Grid göster
            if self.show_grid:
                self._draw_grid()
            
            # OBJ modeli veya basit modeli çiz
            if self.use_obj_model and self.face_display_list and glIsList(self.face_display_list):
                # Model ölçeği ve konumu için ayarlama yap
                glPushMatrix()
                
                # Model boyutunu normalize et (1-2 birim boyut)
                normalize_scale = 2.0 / self.model_size if hasattr(self, 'model_size') else 1.0
                
                # Kullanıcı tarafından ayarlanan ölçek faktörünü uygula
                if hasattr(self, 'user_scale'):
                    normalize_scale *= self.user_scale
                
                # OBJ modeli ölçeklendir - model boyutuna göre ayarla
                glScalef(normalize_scale, normalize_scale, normalize_scale)
                
                # Modeli merkezle - 0,0,0'a getir
                if hasattr(self, 'model_center'):
                    cx, cy, cz = self.model_center
                    glTranslatef(-cx, -cy, -cz)
                    
                try:
                    # Display list ile OBJ modelini çiz
                    glCallList(self.face_display_list)
                except Exception as e:
                    logger.error(f"Error calling display list: {str(e)}")
                    # Basit küre çiz
                    glColor3f(0.9, 0.8, 0.7)  # Ten rengi
                    glutSolidSphere(1.0, 16, 16)
                
                glPopMatrix()
            else:
                # Yüz modeli bileşenlerini çiz (geri dönüş mekanizması)
                for part_name, part in self.face_model.items():
                    if part['type'] == 'sphere':
                        self._draw_sphere(part)
                    elif part['type'] == 'cylinder':
                        self._draw_cylinder(part)
            
            # Debug bilgilerini göster eğer etkinse
            if hasattr(self, 'show_debug_info') and self.show_debug_info:
                self._draw_debug_info()
                
        except Exception as e:
            logger.error(f"Error in paintGL: {str(e)}")
            # GL hatasını temizle
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    def _draw_axes(self):
        """Koordinat eksenlerini ve etiketlerini çiz."""
        try:
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
            
            # Varsayılan değerlere geri dön
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)
        except Exception as e:
            logger.error(f"Error drawing axes: {str(e)}")
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
            
            # MediaPipe açılarını OpenGL'e uygun hale getir
            # İşaret doğrultmaları - orijinal haliyle bize bakması için
            if hasattr(self, 'model_flip') and self.model_flip:
                # Model döndürüldüğünde işaretler tersine çevrilir
                pitch = pitch
                yaw = yaw
                roll = -roll  # roll'un işareti değişir
            else:
                # Normalde işaretlerin ayarlanması 
                pitch = -pitch  # Pitch işareti tersine çevrilir
                yaw = yaw      # Yaw işaretini korunuyor - ayna efektini düzeltmek için
                # roll işareti korunuyor
            
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
            # Eğer rendering aktif değilse, animasyonu durdur
            if not self.rendering_active:
                self.animation_timer.stop()
                return
        
            # Animasyon hız faktörü - gecikmeyi azaltmak için 0.3'ten 0.5'e yükselt
            # Daha hızlı tepki için 0.7'ye yükselt
            speed = 0.7
            
            # Yeni değerleri yumuşak geçişle hesapla
            self.pitch += (self.target_pitch - self.pitch) * speed
            self.yaw += (self.target_yaw - self.yaw) * speed
            self.roll += (self.target_roll - self.roll) * speed
            
            # Son değerleri güncelle
            self._last_pitch = self.pitch
            self._last_yaw = self.yaw
            self._last_roll = self.roll
            
            # Performans optimizasyonu: Sadece yeterli değişiklik varsa güncelle
            # Bu, gereksiz yeniden çizimleri azaltır
            if (abs(self.target_pitch - self.pitch) > 0.05 or
                abs(self.target_yaw - self.yaw) > 0.05 or
                abs(self.target_roll - self.roll) > 0.05):
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
        """Kamera konumunu sıfırla"""
        # Pozisyon ve rotasyonu sıfırla
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.x_trans = 0.0
        self.y_trans = 0.0
        self.z_trans = -5.0
        
        # Model pozisyonunu optimize et
        if hasattr(self, '_optimize_model_position'):
            self._optimize_model_position()
            
        # Kullanıcı ölçeğini sıfırla
        if hasattr(self, 'user_scale'):
            self.user_scale = 1.0
        
        # Geçmiş değerleri sıfırla
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.roll_history.clear()
        
        # Geçmiş değerlere yeni sıfır değerler ekle
        for _ in range(self.history_size):
            self.pitch_history.append(0.0)
            self.yaw_history.append(0.0)
            self.roll_history.append(0.0)
            
        # Hedef açıları sıfırla
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.target_roll = 0.0
        
        # Güncelleme yap
        self.update()
        logger.debug("3D model view reset")

    def _draw_debug_info(self):
        """Debug bilgilerini sağ üst köşede sabit bir konumda göster"""
        glDisable(GL_LIGHTING)
        
        # Ortogonal projeksiyon kullan - 2D çizim için
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        
        # Ekran boyutlarını kullan
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        
        # Viewport koordinatları kullan
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # 2D yüzey için derinlik testini devre dışı bırak
        glDisable(GL_DEPTH_TEST)
        
        # Panel boyutları
        margin = 10  # Kenar boşluğu
        line_height = 18  # Satır yüksekliği
        line_count = 3  # Toplam satır sayısı (sadece açı değerleri)
        box_width = 120  # Kutu genişliği - biraz daraltıldı
        box_height = line_count * line_height + margin * 2  # Kutu yüksekliği
        
        # Sağ üst köşe pozisyonu hesapla
        right_x = self.width() - box_width - margin
        
        # Tamamen opak siyah arka plan kutusu çiz
        glColor4f(0.0, 0.0, 0.0, 1.0)  # Siyah, %100 opak
        
        glBegin(GL_QUADS)
        glVertex2f(right_x, margin)
        glVertex2f(right_x + box_width, margin)
        glVertex2f(right_x + box_width, margin + box_height)
        glVertex2f(right_x, margin + box_height)
        glEnd()
        
        # Debug metinlerini çiz
        glColor3f(1.0, 0.8, 0.2)  # Amber rengi metin
        
        x = right_x + 10  # Sağ kenara göre metin konumu
        y = margin + line_height  # Üst kenardan metin konumu
        
        # Güncel açı değerlerini göster
        self._draw_text_2d(f"Pitch: {self.pitch:.1f}°", x, y)
        y += line_height
        self._draw_text_2d(f"Yaw: {self.yaw:.1f}°", x, y)
        y += line_height
        self._draw_text_2d(f"Roll: {self.roll:.1f}°", x, y)
        
        # 3D görünüme geri dön
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        glEnable(GL_LIGHTING)
    
    def _draw_text_2d(self, text, x, y):
        """
        2D ekran koordinatlarında metin çiz.
        
        Args:
            text: Gösterilecek metin
            x, y: Ekran koordinatları
        """
        glRasterPos2f(x, y)
        
        # Sadece GLUT başlatıldıysa metin göster
        if hasattr(self, 'glut_initialized') and self.glut_initialized:
            try:
                for c in text:
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(c))
            except Exception as e:
                logger.debug(f"Error rendering text with GLUT: {str(e)}")

    def _load_texture(self):
        """
        Doku dosyasını yükler ve OpenGL dokusuna dönüştürür.
        
        Returns:
            int: Doku ID'si
        """
        try:
            if not os.path.exists(self.face_texture_path):
                logger.error(f"Texture file not found: {self.face_texture_path}")
                return None
                
            # PIL ile doku dosyasını yükle
            texture_image = Image.open(self.face_texture_path)
            
            # Boyutu 2^n x 2^m olacak şekilde ayarla (OpenGL için optimal)
            width, height = texture_image.size
            width_2n = 2 ** (width - 1).bit_length()  # En yakın 2^n değeri
            height_2m = 2 ** (height - 1).bit_length()  # En yakın 2^m değeri
            
            # Boyut ayarlaması gerekiyorsa yap
            if width != width_2n or height != height_2m:
                texture_image = texture_image.resize((width_2n, height_2m), Image.LANCZOS)
                logger.info(f"Texture resized from {width}x{height} to {width_2n}x{height_2m}")
            
            # OpenGL için ters çevir (Y ekseni)
            texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # RGBA formatına dönüştür
            texture_data = texture_image.convert("RGBA").tobytes()
            
            # OpenGL dokusunu oluştur
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Doku parametrelerini ayarla
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Dokuyu yükle
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA,
                texture_image.width, texture_image.height,
                0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data
            )
            
            logger.info(f"Texture loaded: {self.face_texture_path}")
            return texture_id
        except Exception as e:
            logger.error(f"Error loading texture: {str(e)}")
            return None
    
    def _create_obj_display_list(self):
        """
        OBJ modeli için display list oluşturur. Bu, render performansını artırır.
        
        Returns:
            int: Display list ID'si
        """
        if not self.face_obj:
            return None
            
        # Yeni display list oluştur
        display_list = glGenLists(1)
        glNewList(display_list, GL_COMPILE)
        
        try:
            # Işığı ve malzeme özelliklerini ayarla
            glEnable(GL_LIGHTING)
            
            # Ten rengi için malzeme ayarla - daha canlı bir ten tonu için güncellenmiş değerler
            ambient = [0.3, 0.25, 0.2, 1.0]
            diffuse = [0.9, 0.8, 0.7, 1.0]  # Ten rengi
            specular = [0.2, 0.2, 0.2, 1.0]  # Hafif parlaklık
            
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 15.0)
            
            # Rengi ayarla
            glColor3f(diffuse[0], diffuse[1], diffuse[2])
            
            # Performans optimizasyonu: Vertex ve face cache'leri
            vertices = self.face_obj.vertices
            
            # Tüm meshler için
            for name, mesh in self.face_obj.meshes.items():
                # Performans optimizasyonu: Eğer mesh büyükse, çizimleri grupla
                batch_size = 200  # Çizim parti boyutu
                total_faces = len(mesh.faces)
                
                for batch_start in range(0, total_faces, batch_size):
                    batch_end = min(batch_start + batch_size, total_faces)
                    
                    # OpenGL'in verimli çizim modlarını kullan
                    glBegin(GL_TRIANGLES)
                    
                    # Bu parti için tüm yüzleri işle
                    for i in range(batch_start, batch_end):
                        face = mesh.faces[i]
                        
                        if len(face) >= 3:
                            # Yüz için normal vektörü hesapla
                            v0 = vertices[face[0]]
                            v1 = vertices[face[1]]
                            v2 = vertices[face[2]]
                            
                            # Performans: Sadece gerekli hesaplamaları yap
                            # İki kenarı hesapla
                            u = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]]
                            v = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]]
                            
                            # Çapraz çarpım ile normal vektörü hesapla
                            nx = u[1] * v[2] - u[2] * v[1]
                            ny = u[2] * v[0] - u[0] * v[2]
                            nz = u[0] * v[1] - u[1] * v[0]
                            
                            # Normal vektörü normalize et
                            length = math.sqrt(nx*nx + ny*ny + nz*nz)
                            if length > 0.0001:  # Sıfıra bölme hatasını önle
                                nx /= length
                                ny /= length
                                nz /= length
                            else:
                                # Geçersiz normal vektörü, varsayılan değer kullan
                                nx, ny, nz = 0.0, 1.0, 0.0
                            
                            # Normal vektörü ayarla - yüz için tek bir normal yeterli
                            glNormal3f(nx, ny, nz)
                            
                            # Yüzün her köşesi için
                            for j in range(len(face)):
                                # Vertex'i çiz
                                v = vertices[face[j]]
                                glVertex3f(v[0], v[1], v[2])
                            
                    glEnd()
                
                # Performans izleme
                logger.debug(f"Rendered mesh '{name}' with {total_faces} faces")
            
        except Exception as e:
            logger.error(f"Error in display list creation: {str(e)}")
            # Hata durumunda basit bir küre çiz
            glColor3f(0.9, 0.8, 0.7)  # Ten rengi
            glutSolidSphere(1.0, 16, 16)
            
        glEndList()
        return display_list

    def set_user_scale(self, scale_value):
        """
        Kullanıcı tarafından ayarlanan ölçek değerini günceller.
        
        Args:
            scale_value: Ölçek faktörü (0.5 - 2.0 arası)
        """
        try:
            self.user_scale = max(0.5, min(2.0, scale_value))
            self.update()  # Widget'ı yeniden çiz
            logger.debug(f"User scale updated: {self.user_scale}")
        except Exception as e:
            logger.error(f"Error setting user scale: {str(e)}")
            
    def toggle_model_orientation(self, flip):
        """
        Modelin yönünü değiştirir (gerçek baş hareketlerini doğrudan takip etmesi için).
        
        Args:
            flip: True ise model 180 derece döndürülür, False ise normal konumunda kalır
        """
        self.model_flip = flip
        self.update()  # Widget'ı yeniden çiz
        logger.debug(f"Model orientation flip: {self.model_flip}")


class Head3DPanel(QWidget):
    """
    3D yüz modeli widget'ı.
    """
    
    def __init__(self, config, parent=None):
        """
        3D yüz modeli panelini oluştur.
        
        Args:
            config: Yapılandırma ayarları
            parent: Ebeveyn widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Panel arka planını siyah olarak ayarla - 3D model daha iyi görünür
        self.setStyleSheet("""
            background-color: black;
            border: none;
            padding: 0px;
        """)
        
        # Layout ekle
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Dış boşlukları tamamen kaldır
        layout.setSpacing(0)  # Panel içi boşlukları da kaldır

        # 3D model görüntüleme widget'ı
        self.head_pose_widget = HeadPoseModelWidget(self, config)
        # OBJ modeli her zaman kullanacak şekilde ayarla
        if hasattr(self.head_pose_widget, 'use_obj_model'):
            self.head_pose_widget.use_obj_model = True
            
        # Widget'ın boyut politikasını genişleyen olarak ayarla
        self.head_pose_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Minimum boyutu küçült, layout'ın genişlik dağılımına daha iyi uyum sağlaması için
        self.head_pose_widget.setMinimumSize(300, 300)
        layout.addWidget(self.head_pose_widget)
        
        # Reset butonu kaldırıldı - artık sağ taraftaki kontrol panelinde
        
        # Varsayılan ölçek değeri
        if not hasattr(self.head_pose_widget, 'user_scale'):
            self.head_pose_widget.user_scale = 1.0
        
        self.setLayout(layout)
        
        logger.debug("Head3DPanel initialized")
    
    def reset_view(self):
        """Kamera konumunu sıfırla"""
        if self.head_pose_widget:
            self.head_pose_widget.reset_view()
    
    def update_pose(self, pitch, yaw, roll):
        """
        Head pose açılarını güncelle.
        
        Args:
            pitch: X-ekseni etrafında dönüş (derece)
            yaw: Y-ekseni etrafında dönüş (derece)
            roll: Z-ekseni etrafında dönüş (derece)
        """
        if self.head_pose_widget:
            self.head_pose_widget.update_pose(pitch, yaw, roll)
            
    def set_user_scale(self, scale_value):
        """Model ölçeğini ayarla."""
        if hasattr(self.head_pose_widget, 'set_user_scale'):
            self.head_pose_widget.set_user_scale(scale_value)
            
    def toggle_model_orientation(self, checked):
        """Model yönünü değiştir."""
        if hasattr(self.head_pose_widget, 'toggle_model_orientation'):
            self.head_pose_widget.toggle_model_orientation(checked)
            
    def set_debug_info(self, checked):
        """Debug bilgilerini göster/gizle."""
        if self.head_pose_widget:
            self.head_pose_widget.show_debug_info = checked
            self.head_pose_widget.update()
            
    def set_axes_visible(self, checked):
        """Eksenleri göster/gizle."""
        if hasattr(self.head_pose_widget, 'axes_enabled'):
            self.head_pose_widget.axes_enabled = checked
            self.head_pose_widget.update()
            
    def set_grid_visible(self, checked):
        """Grid göster/gizle."""
        if hasattr(self.head_pose_widget, 'show_grid'):
            self.head_pose_widget.show_grid = checked
            self.head_pose_widget.update()