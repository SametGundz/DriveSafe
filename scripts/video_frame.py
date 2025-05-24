import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time

class VideoFrameNavigator:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Navigator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Video variables
        self.cap = None
        self.video_path = ""
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.is_playing = False
        self.play_thread = None
        
        # GUI variables
        self.photo = None
        
        self.setup_gui()
    
    def setup_gui(self):
        # Ana frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Başlık
        title_label = tk.Label(
            main_frame, 
            text="Video Frame Navigator", 
            font=("Arial", 20, "bold"),
            fg='white', bg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Video seçim butonu
        self.select_button = tk.Button(
            main_frame,
            text="Video Dosyası Seç",
            command=self.select_video,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.select_button.pack(pady=(0, 20))
        
        # Video görüntüleme alanı
        self.video_frame = tk.Frame(main_frame, bg='black', relief=tk.SUNKEN, bd=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.video_label = tk.Label(
            self.video_frame, 
            text="Video seçmek için yukarıdaki butona tıklayın",
            font=("Arial", 14),
            fg='white', bg='black'
        )
        self.video_label.pack(expand=True)
        
        # Kontrol paneli
        control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        control_frame.pack_propagate(False)
        control_frame.configure(height=120)
        
        # Frame bilgi etiketi
        self.info_label = tk.Label(
            control_frame,
            text="Frame: 0 / 0 | Zaman: 00:00 / 00:00",
            font=("Arial", 10),
            fg='white', bg='#34495e'
        )
        self.info_label.pack(pady=5)
        
        # Frame kaydırma çubuğu
        slider_frame = tk.Frame(control_frame, bg='#34495e')
        slider_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(slider_frame, text="Frame:", fg='white', bg='#34495e').pack(side=tk.LEFT)
        
        self.frame_scale = tk.Scale(
            slider_frame,
            from_=0, to=100,
            orient=tk.HORIZONTAL,
            command=self.on_frame_change,
            bg='#34495e',
            fg='white',
            highlightthickness=0,
            troughcolor='#2c3e50',
            activebackground='#3498db'
        )
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Kontrol butonları
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(pady=10)
        
        # Önceki frame butonu
        self.prev_button = tk.Button(
            button_frame,
            text="⏮ Önceki",
            command=self.previous_frame,
            state=tk.DISABLED,
            bg='#95a5a6',
            fg='white',
            padx=10,
            relief=tk.FLAT
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        # Oynat/Durdur butonu
        self.play_button = tk.Button(
            button_frame,
            text="▶ Oynat",
            command=self.toggle_play,
            state=tk.DISABLED,
            bg='#27ae60',
            fg='white',
            padx=15,
            relief=tk.FLAT
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Sonraki frame butonu
        self.next_button = tk.Button(
            button_frame,
            text="Sonraki ⏭",
            command=self.next_frame,
            state=tk.DISABLED,
            bg='#95a5a6',
            fg='white',
            padx=10,
            relief=tk.FLAT
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Başa dön butonu
        self.reset_button = tk.Button(
            button_frame,
            text="🔄 Başa Dön",
            command=self.reset_video,
            state=tk.DISABLED,
            bg='#e74c3c',
            fg='white',
            padx=10,
            relief=tk.FLAT
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seçin",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path):
        try:
            # Önceki video varsa kapat
            if self.cap:
                self.cap.release()
            
            # Yeni video aç
            self.cap = cv2.VideoCapture(video_path)
            self.video_path = video_path
            
            # Video özelliklerini al
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.current_frame = 0
            
            # GUI'yi güncelle
            self.frame_scale.configure(to=self.total_frames - 1)
            self.frame_scale.set(0)
            
            # Butonları etkinleştir
            self.play_button.configure(state=tk.NORMAL, bg='#27ae60')
            self.next_button.configure(state=tk.NORMAL, bg='#3498db')
            self.prev_button.configure(state=tk.NORMAL, bg='#3498db')
            self.reset_button.configure(state=tk.NORMAL, bg='#e74c3c')
            
            # İlk frame'i göster
            self.show_frame(0)
            self.update_info()
            
            messagebox.showinfo("Başarılı", f"Video yüklendi!\nToplam Frame: {self.total_frames}\nFPS: {self.fps:.2f}")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Video yüklenirken hata oluştu:\n{str(e)}")
    
    def show_frame(self, frame_number):
        if not self.cap:
            return
        
        # Frame'e git
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            # BGR'den RGB'ye dönüştür
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Görüntüyü yeniden boyutlandır
            height, width = frame_rgb.shape[:2]
            display_width = self.video_frame.winfo_width() - 20
            display_height = self.video_frame.winfo_height() - 20
            
            if display_width > 1 and display_height > 1:
                # Oranı koru
                ratio = min(display_width/width, display_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                # PIL Image'e dönüştür ve yeniden boyutlandır
                pil_image = Image.fromarray(frame_rgb)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # PhotoImage'e dönüştür
                self.photo = ImageTk.PhotoImage(pil_image)
                
                # Label'ı güncelle
                self.video_label.configure(image=self.photo, text="")
            
            self.current_frame = frame_number
    
    def on_frame_change(self, value):
        frame_number = int(value)
        self.show_frame(frame_number)
        self.update_info()
    
    def previous_frame(self):
        if self.current_frame > 0:
            new_frame = self.current_frame - 1
            self.frame_scale.set(new_frame)
            self.show_frame(new_frame)
            self.update_info()
    
    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            new_frame = self.current_frame + 1
            self.frame_scale.set(new_frame)
            self.show_frame(new_frame)
            self.update_info()
    
    def reset_video(self):
        self.is_playing = False
        self.play_button.configure(text="▶ Oynat")
        self.frame_scale.set(0)
        self.show_frame(0)
        self.update_info()
    
    def toggle_play(self):
        if not self.cap:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.configure(text="⏸ Durdur")
            self.start_playback()
        else:
            self.play_button.configure(text="▶ Oynat")
    
    def start_playback(self):
        if self.play_thread and self.play_thread.is_alive():
            return
        
        self.play_thread = threading.Thread(target=self.play_video)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def play_video(self):
        frame_delay = 1.0 / self.fps
        
        while self.is_playing and self.current_frame < self.total_frames - 1:
            start_time = time.time()
            
            # Sonraki frame'e geç
            next_frame = self.current_frame + 1
            self.root.after(0, lambda: self.frame_scale.set(next_frame))
            self.root.after(0, lambda: self.show_frame(next_frame))
            self.root.after(0, self.update_info)
            
            # FPS'e uygun bekleme
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
        
        # Video bittiyse oynatmayı durdur
        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.root.after(0, lambda: self.play_button.configure(text="▶ Oynat"))
    
    def update_info(self):
        if self.cap:
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps
            
            current_time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            total_time_str = f"{int(total_time//60):02d}:{int(total_time%60):02d}"
            
            info_text = f"Frame: {self.current_frame + 1} / {self.total_frames} | Zaman: {current_time_str} / {total_time_str}"
            self.info_label.configure(text=info_text)
    
    def __del__(self):
        if self.cap:
            self.cap.release()

def main():
    root = tk.Tk()
    app = VideoFrameNavigator(root)
    
    # Pencere kapatılırken temizlik yap
    def on_closing():
        if app.cap:
            app.cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()