#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph utilities module for the driver drowsiness detection application.

This module implements visualization components, particularly the expanded charts window
and related functionality for displaying and saving metrics graphs.
"""

import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QPixmap
from PyQt6.QtCharts import QChart, QLineSeries, QValueAxis, QChartView


class ExpandedChartsWindow(QMainWindow):
    """
    Genişletilmiş grafikleri gösteren ayrı pencere.
    
    Bu pencere, EAR, MAR ve PERCLOS değerlerini ayrı ayrı,
    daha büyük grafikler halinde gösterir.
    """
    
    def __init__(self, config, parent=None):
        """
        Genişletilmiş grafikler penceresini başlat.
        
        Args:
            config: Yapılandırma sözlüğü
            parent: Ebeveyn pencere
        """
        super().__init__(parent)
        
        self.config = config
        self.parent_window = parent
        
        # Pencere özelliklerini ayarla
        self.setWindowTitle("Genişletilmiş Grafikler")
        # Daha büyük bir pencere boyutu ayarla
        self.resize(1200, 900)
        
        # Ana widget ve layout oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Her bir metrik için ayrı bir grafik oluştur
        self._create_ear_chart(main_layout)
        self._create_mar_chart(main_layout)
        self._create_perclos_chart(main_layout)
        
        # Bilimsel analiz için ek özellikler
        control_layout = QHBoxLayout()
        
        # X eksenini ayarlama
        x_range_layout = QHBoxLayout()
        x_range_layout.addWidget(QLabel("X Ekseni Süresi (sn):"))
        
        self.x_range_combo = QComboBox()
        self.x_range_combo.addItems(["10", "30", "60", "120", "300", "600"])
        self.x_range_combo.setCurrentText(str(self.config['chart']['history_duration']))
        self.x_range_combo.currentTextChanged.connect(self._update_x_range)
        x_range_layout.addWidget(self.x_range_combo)
        
        control_layout.addLayout(x_range_layout)
        control_layout.addStretch()
        
        # Grafikleri kaydetme butonu
        save_button = QPushButton("Grafikleri Kaydet")
        save_button.setMinimumWidth(150)  # Buton genişliğini artır
        save_button.setMinimumHeight(40)  # Buton yüksekliğini artır
        save_button.setFont(QFont(self.config['fonts']['family'], 11, QFont.Weight.Bold))  # Yazı tipini büyüt
        save_button.clicked.connect(self._save_charts)
        control_layout.addWidget(save_button)
        
        main_layout.addLayout(control_layout)
    
    def _create_ear_chart(self, parent_layout):
        """EAR grafiğini oluştur."""
        chart = QChart()
        chart.setTitle("EAR (Göz Açıklık Oranı) Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        
        # Daha büyük başlık yazı tipi
        title_font = QFont(self.config['fonts']['family'], 14, QFont.Weight.Bold)
        chart.setTitleFont(title_font)
        
        # Series oluştur
        self.ear_series = QLineSeries()
        self.ear_series.setName("EAR")
        self.ear_series.setPen(QPen(QColor("#007aff"), 3, Qt.PenStyle.SolidLine))  # Çizgi kalınlığını artır
        self.ear_series.setUseOpenGL(True)
        chart.addSeries(self.ear_series)
        
        # X ekseni
        self.ear_time_axis = QValueAxis()
        self.ear_time_axis.setRange(0, self.config['chart']['history_duration'])
        self.ear_time_axis.setTitleText("Zaman (sn)")
        # Eksen yazı tipi boyutunu artır
        axis_font = QFont(self.config['fonts']['family'], 12)
        self.ear_time_axis.setTitleFont(axis_font)
        self.ear_time_axis.setLabelsFont(QFont(self.config['fonts']['family'], 10))
        chart.addAxis(self.ear_time_axis, Qt.AlignmentFlag.AlignBottom)
        self.ear_series.attachAxis(self.ear_time_axis)
        
        # Y ekseni
        self.ear_value_axis = QValueAxis()
        self.ear_value_axis.setRange(
            self.config['chart']['y_range_ear'][0],
            self.config['chart']['y_range_ear'][1]
        )
        self.ear_value_axis.setTitleText("EAR Değeri")
        self.ear_value_axis.setTitleFont(axis_font)
        self.ear_value_axis.setLabelsFont(QFont(self.config['fonts']['family'], 10))
        
        # EAR eşiğini gösterme çizgisi ekle
        ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
        ear_line = QLineSeries()
        ear_line.setName("EAR Eşiği")
        ear_line.setPen(QPen(QColor("#ff3b30"), 2, Qt.PenStyle.DashLine))  # Çizgi kalınlığını artır
        ear_line.append(0, ear_threshold)
        ear_line.append(self.config['chart']['history_duration'], ear_threshold)
        chart.addSeries(ear_line)
        chart.addAxis(self.ear_value_axis, Qt.AlignmentFlag.AlignLeft)
        self.ear_series.attachAxis(self.ear_value_axis)
        ear_line.attachAxis(self.ear_time_axis)
        ear_line.attachAxis(self.ear_value_axis)
        
        # Grafik görünümü oluştur
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        parent_layout.addWidget(chart_view)
    
    def _create_mar_chart(self, parent_layout):
        """MAR grafiğini oluştur."""
        chart = QChart()
        chart.setTitle("MAR (Ağız Açıklık Oranı) Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        
        # Daha büyük başlık yazı tipi
        title_font = QFont(self.config['fonts']['family'], 14, QFont.Weight.Bold)
        chart.setTitleFont(title_font)
        
        # Series oluştur
        self.mar_series = QLineSeries()
        self.mar_series.setName("MAR")
        self.mar_series.setPen(QPen(QColor("#5ac8fa"), 3, Qt.PenStyle.SolidLine))  # Çizgi kalınlığını artır
        self.mar_series.setUseOpenGL(True)
        chart.addSeries(self.mar_series)
        
        # X ekseni
        self.mar_time_axis = QValueAxis()
        self.mar_time_axis.setRange(0, self.config['chart']['history_duration'])
        self.mar_time_axis.setTitleText("Zaman (sn)")
        # Eksen yazı tipi boyutunu artır
        axis_font = QFont(self.config['fonts']['family'], 12) 
        self.mar_time_axis.setTitleFont(axis_font)
        self.mar_time_axis.setLabelsFont(QFont(self.config['fonts']['family'], 10))
        chart.addAxis(self.mar_time_axis, Qt.AlignmentFlag.AlignBottom)
        self.mar_series.attachAxis(self.mar_time_axis)
        
        # Y ekseni
        self.mar_value_axis = QValueAxis()
        self.mar_value_axis.setRange(
            self.config['chart']['y_range_mar'][0],
            self.config['chart']['y_range_mar'][1]
        )
        self.mar_value_axis.setTitleText("MAR Değeri")
        self.mar_value_axis.setTitleFont(axis_font)
        self.mar_value_axis.setLabelsFont(QFont(self.config['fonts']['family'], 10))
        chart.addAxis(self.mar_value_axis, Qt.AlignmentFlag.AlignLeft)
        self.mar_series.attachAxis(self.mar_value_axis)
        
        # Grafik görünümü oluştur
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        parent_layout.addWidget(chart_view)
    
    def _create_perclos_chart(self, parent_layout):
        """PERCLOS grafiğini oluştur."""
        chart = QChart()
        chart.setTitle("PERCLOS (Göz Kapanma Yüzdesi) Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        
        # Daha büyük başlık yazı tipi
        title_font = QFont(self.config['fonts']['family'], 14, QFont.Weight.Bold)
        chart.setTitleFont(title_font)
        
        # Series oluştur
        self.perclos_series = QLineSeries()
        self.perclos_series.setName("PERCLOS")
        self.perclos_series.setPen(QPen(QColor("#ff9500"), 3, Qt.PenStyle.SolidLine))  # Çizgi kalınlığını artır
        self.perclos_series.setUseOpenGL(True)
        chart.addSeries(self.perclos_series)
        
        # X ekseni
        self.perclos_time_axis = QValueAxis()
        self.perclos_time_axis.setRange(0, self.config['chart']['history_duration'])
        self.perclos_time_axis.setTitleText("Zaman (sn)")
        # Eksen yazı tipi boyutunu artır
        axis_font = QFont(self.config['fonts']['family'], 12)
        self.perclos_time_axis.setTitleFont(axis_font)
        self.perclos_time_axis.setLabelsFont(QFont(self.config['fonts']['family'], 10))
        chart.addAxis(self.perclos_time_axis, Qt.AlignmentFlag.AlignBottom)
        self.perclos_series.attachAxis(self.perclos_time_axis)
        
        # Y ekseni
        self.perclos_value_axis = QValueAxis()
        self.perclos_value_axis.setRange(
            self.config['chart']['y_range_perclos'][0],
            self.config['chart']['y_range_perclos'][1]
        )
        self.perclos_value_axis.setTitleText("PERCLOS Değeri (%)")
        self.perclos_value_axis.setTitleFont(axis_font)
        self.perclos_value_axis.setLabelsFont(QFont(self.config['fonts']['family'], 10))
        
        # PERCLOS eşiğini gösterme çizgisi ekle
        perclos_threshold = self.config.get('indicators', {}).get('perclos', {}).get('critical_threshold', 20.0)
        perclos_line = QLineSeries()
        perclos_line.setName("PERCLOS Eşiği")
        perclos_line.setPen(QPen(QColor("#ff3b30"), 2, Qt.PenStyle.DashLine))  # Çizgi kalınlığını artır
        perclos_line.append(0, perclos_threshold)
        perclos_line.append(self.config['chart']['history_duration'], perclos_threshold)
        chart.addSeries(perclos_line)
        chart.addAxis(self.perclos_value_axis, Qt.AlignmentFlag.AlignLeft)
        self.perclos_series.attachAxis(self.perclos_value_axis)
        perclos_line.attachAxis(self.perclos_time_axis)
        perclos_line.attachAxis(self.perclos_value_axis)
        
        # Grafik görünümü oluştur
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        parent_layout.addWidget(chart_view)
    
    def initialize_with_data(self, ear_data, mar_data, perclos_data):
        """Grafikleri mevcut verilerle başlat."""
        # Mevcut serileri temizle
        self.ear_series.clear()
        self.mar_series.clear()
        self.perclos_series.clear()
        
        # EAR verisini ekle
        for x, y in ear_data:
            self.ear_series.append(x, y)
        
        # MAR verisini ekle
        for x, y in mar_data:
            self.mar_series.append(x, y)
        
        # PERCLOS verisini ekle
        for x, y in perclos_data:
            self.perclos_series.append(x, y)
    
    def update_charts(self):
        """Ana penceredeki verilerle grafikleri güncelle."""
        if not self.parent_window.is_capturing:
            return
        
        # Ana pencereden yeni veri noktaları al
        time = self.parent_window.time_counter
        ear = self.parent_window.metrics_panel.ear_indicator.last_value
        mar = self.parent_window.metrics_panel.mar_indicator.last_value
        perclos = self.parent_window.metrics_panel.perclos_indicator.last_value
        
        # Verileri grafiklere ekle
        self.ear_series.append(time, ear)
        self.mar_series.append(time, mar)
        self.perclos_series.append(time, perclos)
        
        # X ekseni aralığını güncelle
        history_duration = float(self.x_range_combo.currentText())
        if time > history_duration:
            # Eğer zaman tarih aralığını aştıysa, eski noktaları kaldır
            cutoff_time = time - history_duration
            
            # EAR serisinden eski noktaları kaldır
            while self.ear_series.count() > 0 and self.ear_series.at(0).x() < cutoff_time:
                self.ear_series.remove(0)
            
            # MAR serisinden eski noktaları kaldır
            while self.mar_series.count() > 0 and self.mar_series.at(0).x() < cutoff_time:
                self.mar_series.remove(0)
            
            # PERCLOS serisinden eski noktaları kaldır
            while self.perclos_series.count() > 0 and self.perclos_series.at(0).x() < cutoff_time:
                self.perclos_series.remove(0)
            
            # Zaman eksenlerini güncelle
            self.ear_time_axis.setRange(time - history_duration, time)
            self.mar_time_axis.setRange(time - history_duration, time)
            self.perclos_time_axis.setRange(time - history_duration, time)
    
    def _update_x_range(self, value):
        """X ekseninin zaman aralığını güncelle."""
        history_duration = float(value)
        
        # Ana pencere zaman aralığını değiştirme (isteğe bağlı)
        # self.parent_window.config['chart']['history_duration'] = history_duration
        
        # Mevcut zamandan itibaren görünüm aralığını ayarla
        current_time = self.parent_window.time_counter
        start_time = max(0, current_time - history_duration)
        
        # Eksenleri güncelle
        self.ear_time_axis.setRange(start_time, current_time)
        self.mar_time_axis.setRange(start_time, current_time)
        self.perclos_time_axis.setRange(start_time, current_time)
    
    def _save_charts(self):
        """Grafikleri görüntü dosyaları olarak kaydet."""
        from datetime import datetime
        import os
        
        # Zaman damgası oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Proje kök dizinini bul ve kayıt klasörü oluştur
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        save_dir = os.path.join(root_dir, "recordings", "stats", timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        # Her bir grafiği kaydet
        try:
            # Ana widget'taki tüm grafikler
            chart_views = self.findChildren(QChartView)
            
            for i, chart_view in enumerate(chart_views):
                metric_name = ["EAR", "MAR", "PERCLOS"][i]
                filename = os.path.join(save_dir, f"{metric_name}.png")
                
                # Grafiği yüksek çözünürlüklü görüntü olarak kaydet
                pixmap = QPixmap(chart_view.size())
                pixmap.fill(Qt.GlobalColor.white)
                
                painter = QPainter(pixmap)
                chart_view.render(painter)
                painter.end()
                
                pixmap.save(filename)
            
            # Basit bir onay mesajı göster
            QMessageBox.information(
                self,
                "Bilgi",
                f"Grafikler kaydedildi:\n/recordings/stats/{timestamp}",
                QMessageBox.StandardButton.Ok
            )
            
        except Exception as e:
            # Basit bir hata mesajı göster
            QMessageBox.warning(
                self,
                "Hata",
                f"Grafikler kaydedilirken bir hata oluştu:\n{str(e)}",
                QMessageBox.StandardButton.Ok
            )
    
    def closeEvent(self, event):
        """Pencere kapatıldığında timer bağlantısını kaldır."""
        # Not: Bu metod artık _on_expanded_charts_close tarafından geçersiz kılınıyor
        event.accept()


# Yardımcı fonksiyonlar
def create_chart_data_from_series(series):
    """QLineSeries'ten veri noktalarını liste olarak döndür."""
    data = []
    for i in range(series.count()):
        point = series.at(i)
        data.append((point.x(), point.y()))
    return data


def save_chart_as_image(chart_view, filename):
    """Grafik görünümünü görüntü dosyası olarak kaydet."""
    pixmap = QPixmap(chart_view.size())
    pixmap.fill(Qt.GlobalColor.white)
    
    painter = QPainter(pixmap)
    chart_view.render(painter)
    painter.end()
    
    return pixmap.save(filename) 