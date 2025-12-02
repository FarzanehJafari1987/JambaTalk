import sys
import os
import threading
import time
import queue
import numpy as np
import torch
import sounddevice as sd
import cv2
import pickle
import trimesh
import pyrender
import librosa
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import wave

# Fix Qt plugin crash
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QComboBox, 
                            QFileDialog, QRadioButton, QButtonGroup, 
                            QProgressBar, QGraphicsDropShadowEffect, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from transformers import Wav2Vec2Processor
from jambatalk import JambaTalk

# Audio configuration
SAMPLE_RATE = 16000
WINDOW_DURATION = 1.0
STRIDE_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * 0.03)

# Global queues
audio_queue = queue.Queue()
mesh_queue = queue.Queue(maxsize=2)
playback_queue = queue.Queue()  # Queue for audio playback


class AudioPlaybackThread(QThread):
    """Thread for playing back audio synchronized with video"""
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.stream = None
        
    def run(self):
        """Play audio chunks from the playback queue"""
        self.running = True
        
        def callback(outdata, frames, time_info, status):
            if status:
                print(f"Playback status: {status}")
            
            try:
                data = playback_queue.get_nowait()
                if len(data) < frames:
                    # Pad if necessary
                    data = np.pad(data, (0, frames - len(data)), 'constant')
                outdata[:] = data[:frames].reshape(-1, 1)
            except queue.Empty:
                # No audio available, output silence
                outdata.fill(0)
        
        try:
            self.stream = sd.OutputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                callback=callback
            )
            self.stream.start()
            
            while self.running:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
    
    def stop(self):
        """Stop audio playback"""
        self.running = False
        # Clear the playback queue
        while not playback_queue.empty():
            try:
                playback_queue.get_nowait()
            except queue.Empty:
                break


class AudioFileProcessor(QThread):
    """Thread for processing audio files"""
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, audio_queue, playback_speed=1.0):
        super().__init__()
        self.file_path = file_path
        self.audio_queue = audio_queue
        self.playback_speed = max(0.1, min(5.0, playback_speed))
        self.running = False
        
    def run(self):
        try:
            audio, sr = librosa.load(self.file_path, sr=SAMPLE_RATE)
            total_samples = len(audio)
            
            if total_samples == 0:
                self.error.emit("Empty audio file")
                return
                
            chunk_samples = CHUNK_SIZE
            sleep_time = (chunk_samples / SAMPLE_RATE) / self.playback_speed
            
            self.running = True
            processed_samples = 0
            
            while self.running and processed_samples < total_samples:
                end_idx = min(processed_samples + chunk_samples, total_samples)
                chunk = audio[processed_samples:end_idx]
                
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                
                self.audio_queue.put(chunk)
                processed_samples = end_idx
                
                progress = int((processed_samples / total_samples) * 100)
                self.progress.emit(progress)
                
                time.sleep(sleep_time)
                
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Error processing audio file: {str(e)}")
            
    def stop(self):
        self.running = False


@torch.no_grad()
def extract_features(processor, audio_chunk, device):
    """Extract features from audio chunk and ensure correct dtype"""
    inputs = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    return inputs.input_values.to(device).half()


class RealtimeRenderer:
    """3D mesh renderer with white background"""
    
    def __init__(self, faces, viewport_size=(512, 512)):
        self.faces = faces
        self.viewport_size = viewport_size
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=viewport_size[0], 
            viewport_height=viewport_size[1]
        )
        
        self.scene = pyrender.Scene(
            ambient_light=[0.5, 0.5, 0.5], 
            bg_color=[255, 255, 255]  # White background
        )
        self.mesh_node = None

        # Camera setup
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pose = np.eye(4)
        pose[2, 3] = 0.35
        self.scene.add(cam, pose=pose)
        
        # Lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        self.scene.add(light, pose=pose)

    def update_mesh(self, vertices):
        """Update mesh with new vertices"""
        if self.mesh_node is not None:
            try:
                self.scene.remove_node(self.mesh_node)
            except Exception:
                pass

        # Vibrant gradient color for face mesh
        color = np.array([99, 102, 241], dtype=np.uint8)  # Indigo-500
        colors = np.tile(color, (vertices.shape[0], 1))

        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, vertex_colors=colors)
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        self.mesh_node = self.scene.add(pyrender_mesh)

    def render(self):
        """Render the current scene"""
        try:
            color, _ = self.renderer.render(self.scene)
            return np.ascontiguousarray(color)
        except Exception as e:
            print(f"Render error: {e}")
            return np.zeros((self.viewport_size[1], self.viewport_size[0], 3), dtype=np.uint8)


class VideoRecorder:
    """Handle video and audio recording with proper mixing"""
    
    def __init__(self, output_path, fps=30, frame_size=(512, 512), sample_rate=16000):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.frames = []
        self.audio_samples = []
        self.is_recording = False
        self.lock = threading.Lock()
        
    def start(self):
        """Start recording"""
        with self.lock:
            self.is_recording = True
            self.frames = []
            self.audio_samples = []
        print(f"Recording started: {self.output_path}")
        
    def add_frame(self, frame):
        """Add a video frame"""
        if self.is_recording:
            with self.lock:
                self.frames.append(frame.copy())
            
    def add_audio(self, audio_chunk):
        """Add audio samples"""
        if self.is_recording:
            with self.lock:
                self.audio_samples.append(audio_chunk.copy())
    
    def stop(self):
        """Stop recording and save to file"""
        if not self.is_recording:
            return False
            
        with self.lock:
            self.is_recording = False
            frames_copy = self.frames.copy()
            audio_copy = self.audio_samples.copy()
        
        if len(frames_copy) == 0:
            print("No frames to save")
            return False
        
        try:
            # Create temporary files
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            temp_video = os.path.join(temp_dir, f"temp_video_{timestamp}.mp4")
            temp_audio = os.path.join(temp_dir, f"temp_audio_{timestamp}.wav")
            
            # Save video
            print(f"Saving {len(frames_copy)} frames to video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video, fourcc, self.fps, self.frame_size)
            
            for frame in frames_copy:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            
            video_writer.release()
            print(f"Video saved: {temp_video}")
            
            # Save audio if available
            has_audio = len(audio_copy) > 0
            if has_audio:
                print(f"Saving audio with {len(audio_copy)} chunks...")
                audio_data = np.concatenate(audio_copy)
                
                # Ensure audio length matches video duration
                video_duration = len(frames_copy) / self.fps
                expected_audio_samples = int(video_duration * self.sample_rate)
                
                if len(audio_data) > expected_audio_samples:
                    audio_data = audio_data[:expected_audio_samples]
                elif len(audio_data) < expected_audio_samples:
                    audio_data = np.pad(audio_data, (0, expected_audio_samples - len(audio_data)), 'constant')
                
                with wave.open(temp_audio, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                print(f"Audio saved: {temp_audio}")
            
            # Merge video and audio using ffmpeg
            if has_audio:
                print("Merging video and audio with ffmpeg...")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', temp_audio,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-strict', 'experimental',
                    '-shortest',
                    self.output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    import shutil
                    shutil.copy(temp_video, self.output_path)
                    print(f"Saved video without audio: {self.output_path}")
                else:
                    print(f"Successfully merged video and audio: {self.output_path}")
            else:
                import shutil
                shutil.copy(temp_video, self.output_path)
                print(f"Saved video without audio: {self.output_path}")
            
            # Cleanup
            try:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                if has_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"Error saving recording: {e}")
            import traceback
            traceback.print_exc()
            return False


def add_shadow(widget, blur_radius=20, offset_y=5, color=QColor(0, 0, 0, 50)):
    """Add drop shadow effect to widget"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setXOffset(0)
    shadow.setYOffset(offset_y)
    shadow.setColor(color)
    widget.setGraphicsEffect(shadow)


class JambaTalkApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.running = False
        self.audio_stream = None
        self.audio_file_processor = None
        self.current_audio_file = None
        self.last_infer_time = 0.0
        self.window_fps = 0.0
        self.total_frames = 0
        self.frames_per_window = int(WINDOW_DURATION * args.fps)
        self.recorder = None
        self.is_recording = False
        self.playback_thread = None  # Audio playback thread
        
        self.init_model()
        self.init_ui()
        self.populate_microphones()
        
        # Timers
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        
    def init_model(self):
        """Initialize the JambaTalk model"""
        print("Loading model...")
        self.device = torch.device(self.args.device)
        
        self.model = JambaTalk(self.args).to(self.device)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        self.model = self.model.half()
        
        print("Applying dtype fixes...")
        for name, param in self.model.named_parameters():
            if param.dtype != torch.float16:
                param.data = param.data.half()
        
        for name, buffer in self.model.named_buffers():
            if buffer.dtype != torch.float16:
                buffer.data = buffer.data.half()
        
        original_predict = self.model.predict
        def safe_predict(audio_features, template):
            audio_features = audio_features.half() if audio_features.dtype != torch.float16 else audio_features
            template = template.half() if template.dtype != torch.float16 else template
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return original_predict(audio_features, template)
        
        self.model.predict = safe_predict
        self.model.eval()

        with open(self.args.template_path, 'rb') as f:
            templates = pickle.load(f, encoding='latin1')
        self.template = torch.HalfTensor(templates[self.args.subject].reshape(1, -1)).to(self.device)

        self.faces = trimesh.load(self.args.render_template_path, force='mesh').faces
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.renderer = RealtimeRenderer(self.faces)
        
        print(f"Model loaded successfully!")

    def init_ui(self):
        """Initialize beautiful, modern user interface"""
        self.setWindowTitle("JambaTalk • Real-Time 3D Talking Face")
        self.setGeometry(100, 100, 1000, 750)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ===== LEFT SIDEBAR: Controls =====
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366F1, stop:0.5 #8B5CF6, stop:1 #A855F7);
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(0)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # App Header with modern styling
        header_container = QWidget()
        header_container.setStyleSheet("background: transparent;")
        header_layout = QVBoxLayout(header_container)
        header_layout.setContentsMargins(25, 25, 25, 20)
        
        app_title = QLabel("JambaTalk")
        title_font = QFont("Segoe UI", 24, QFont.Bold)
        title_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.8)
        app_title.setFont(title_font)
        app_title.setStyleSheet("color: white;")
        app_title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(app_title)
        
        subtitle = QLabel("Real-Time 3D Talking Face")
        subtitle_font = QFont("Segoe UI", 9)
        subtitle_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.4)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.85);")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle)
        
        sidebar_layout.addWidget(header_container)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background: rgba(255, 255, 255, 0.2); max-height: 1px;")
        sidebar_layout.addWidget(divider)
        
        # Controls Container
        controls_container = QWidget()
        controls_container.setStyleSheet("background: transparent;")
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(18)
        controls_layout.setContentsMargins(25, 20, 25, 20)
        
        # Input Source Section
        source_section = self.create_section("INPUT SOURCE")
        
        self.input_group = QButtonGroup()
        self.mic_radio = QRadioButton("Microphone")
        self.file_radio = QRadioButton("Audio File")
        self.mic_radio.setChecked(True)
        
        radio_font = QFont("Segoe UI", 10)
        self.mic_radio.setFont(radio_font)
        self.file_radio.setFont(radio_font)
        
        radio_style = """
            QRadioButton {
                color: white;
                padding: 10px 12px;
                spacing: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            QRadioButton:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid rgba(255, 255, 255, 0.6);
                background: transparent;
            }
            QRadioButton::indicator:checked {
                border: 2px solid white;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5, stop:0 white, stop:0.6 white, stop:0.7 transparent);
            }
            QRadioButton::indicator:hover {
                border: 2px solid rgba(255, 255, 255, 0.9);
            }
        """
        self.mic_radio.setStyleSheet(radio_style)
        self.file_radio.setStyleSheet(radio_style)
        
        self.input_group.addButton(self.mic_radio)
        self.input_group.addButton(self.file_radio)
        
        source_section.addWidget(self.mic_radio)
        source_section.addSpacing(6)
        source_section.addWidget(self.file_radio)
        
        controls_layout.addLayout(source_section)
        
        # Device Selection Section
        device_section = self.create_section("AUDIO DEVICE")
        
        self.mic_combo = QComboBox()
        combo_font = QFont("Segoe UI", 10)
        self.mic_combo.setFont(combo_font)
        self.mic_combo.setStyleSheet("""
            QComboBox {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                padding: 14px 18px;
                color: white;
                selection-background-color: rgba(255, 255, 255, 0.3);
            }
            QComboBox:hover {
                background: rgba(255, 255, 255, 0.25);
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QComboBox:focus {
                border: 2px solid white;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 18px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid white;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background: #6366F1;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                selection-background-color: rgba(255, 255, 255, 0.3);
                color: white;
                padding: 8px;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 10px 15px;
                border-radius: 6px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(255, 255, 255, 0.2);
            }
        """)
        device_section.addWidget(self.mic_combo)
        
        controls_layout.addLayout(device_section)
        
        # File Selection Section
        file_section = self.create_section("AUDIO FILE")
        
        self.file_button = QPushButton("Choose Audio File")
        button_font = QFont("Segoe UI", 10, QFont.Medium)
        self.file_button.setFont(button_font)
        self.file_button.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                padding: 12px 15px;
                color: white;
                text-align: center;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.15);
            }
        """)
        self.file_button.clicked.connect(self.select_audio_file)
        file_section.addWidget(self.file_button)
        
        file_section.addSpacing(6)
        
        self.file_label = QLabel("No file selected")
        file_label_font = QFont("Segoe UI", 8)
        self.file_label.setFont(file_label_font)
        self.file_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.7);
            padding: 6px 15px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 6px;
        """)
        self.file_label.setWordWrap(True)
        file_section.addWidget(self.file_label)
        
        controls_layout.addLayout(file_section)
        
        controls_layout.addStretch()
        
        # Action Buttons with enhanced styling
        self.run_button = QPushButton("START")
        action_font = QFont("Segoe UI", 12, QFont.Bold)
        action_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        self.run_button.setFont(action_font)
        self.run_button.setMinimumHeight(50)
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.setStyleSheet("""
            QPushButton {
                background: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                color: #6366F1;
            }
            QPushButton:hover {
                background: #F8F9FF;
            }
            QPushButton:pressed {
                background: #E8E9FF;
            }
        """)
        self.run_button.clicked.connect(self.toggle_processing)
        controls_layout.addWidget(self.run_button)
        
        controls_layout.addSpacing(8)
        
        self.record_button = QPushButton("RECORD")
        self.record_button.setFont(action_font)
        self.record_button.setMinimumHeight(50)
        self.record_button.setCursor(Qt.PointingHandCursor)
        self.record_button.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.15);
                border: 2px solid rgba(255, 255, 255, 0.4);
                border-radius: 12px;
                padding: 15px;
                color: white;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.25);
                border: 2px solid rgba(255, 255, 255, 0.6);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.05);
                border: 2px solid rgba(255, 255, 255, 0.15);
                color: rgba(255, 255, 255, 0.4);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.1);
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        controls_layout.addWidget(self.record_button)
        
        controls_layout.addSpacing(15)
        
        # Status Cards
        self.status_label = QLabel("Ready to begin")
        status_font = QFont("Segoe UI", 9)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("""
            color: white;
            background: rgba(255, 255, 255, 0.15);
            padding: 12px 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        """)
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        
        self.recording_label = QLabel("")
        rec_font = QFont("Segoe UI", 9, QFont.Bold)
        self.recording_label.setFont(rec_font)
        self.recording_label.setStyleSheet("""
            color: #DC2626;
            background: white;
            padding: 12px 15px;
            border-radius: 8px;
        """)
        controls_layout.addWidget(self.recording_label)
        
        sidebar_layout.addWidget(controls_container)
        
        # Footer
        # footer = QLabel("Powered by PyTorch")
        # footer_font = QFont("Segoe UI", 8)
        # footer.setFont(footer_font)
        # footer.setStyleSheet("""
        #     color: rgba(255, 255, 255, 0.5);
        #     padding: 15px;
        # """)
        # footer.setAlignment(Qt.AlignCenter)
        # sidebar_layout.addWidget(footer)
        
        main_layout.addWidget(sidebar)
        
        # ===== RIGHT PANEL: Video Display =====
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #F8FAFC, stop:1 #F1F5F9);
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(30, 30, 30, 30)
        
        # Stats Bar
        stats_bar = QWidget()
        stats_bar.setStyleSheet("""
            background: white;
            border-radius: 12px;
        """)
        stats_layout = QHBoxLayout(stats_bar)
        stats_layout.setContentsMargins(20, 12, 20, 12)
        
        self.fps_label = QLabel("FPS: --")
        self.latency_label = QLabel("Latency: --")
        self.frames_label = QLabel("Frames: 0")
        
        for label in [self.fps_label, self.latency_label, self.frames_label]:
            label.setFont(QFont("Segoe UI", 9, QFont.Medium))
            label.setStyleSheet("color: #64748B;")
            stats_layout.addWidget(label)
            if label != self.frames_label:
                stats_layout.addWidget(self.create_stat_separator())
        
        stats_layout.addStretch()
        right_layout.addWidget(stats_bar)
        
        right_layout.addSpacing(20)
        
        # Video Container
        video_container = QWidget()
        video_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 16px;
            }
        """)
        add_shadow(video_container, blur_radius=25, offset_y=8, color=QColor(0, 0, 0, 30))
        
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(25, 25, 25, 25)
        
        # Video header
        video_header_layout = QHBoxLayout()
        
        preview_icon = QLabel("●")
        preview_icon.setFont(QFont("Segoe UI", 10))
        preview_icon.setStyleSheet("color: #10B981;")
        video_header_layout.addWidget(preview_icon)
        
        video_header = QLabel("LIVE PREVIEW")
        video_header_font = QFont("Segoe UI", 9, QFont.Bold)
        video_header_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        video_header.setFont(video_header_font)
        video_header.setStyleSheet("color: #64748B;")
        video_header_layout.addWidget(video_header)
        
        video_header_layout.addStretch()
        
        self.quality_badge = QLabel("HD")
        self.quality_badge.setFont(QFont("Segoe UI", 7, QFont.Bold))
        self.quality_badge.setStyleSheet("""
            color: white;
            background: #6366F1;
            padding: 3px 8px;
            border-radius: 5px;
        """)
        video_header_layout.addWidget(self.quality_badge)
        
        video_layout.addLayout(video_header_layout)
        
        video_layout.addSpacing(18)
        
        # Video Display with modern styling
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 480)
        video_font = QFont("Segoe UI", 11)
        self.video_label.setFont(video_font)
        self.video_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F8FAFC, stop:1 #EFF6FF);
                border: 2px dashed #CBD5E1;
                border-radius: 14px;
                color: #94A3B8;
                padding: 30px;
            }
        """)
        self.video_label.setText("Click START to begin")
        video_layout.addWidget(self.video_label)
        
        right_layout.addWidget(video_container)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(5)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 2px;
                background: rgba(255, 255, 255, 0.5);
                margin-top: 15px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366F1, stop:0.5 #8B5CF6, stop:1 #A855F7);
                border-radius: 2px;
            }
        """)
        right_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(right_panel, 1)

    def create_section(self, title):
        """Create a styled section with title"""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        label = QLabel(title)
        label_font = QFont("Segoe UI", 9, QFont.Bold)
        label_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        label.setFont(label_font)
        label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
        layout.addWidget(label)
        
        return layout
    
    def create_stat_separator(self):
        """Create a vertical separator for stats"""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("background: #E2E8F0; max-width: 1px;")
        separator.setFixedHeight(20)
        return separator

    def populate_microphones(self):
        """Populate microphone dropdown"""
        try:
            devices = sd.query_devices()
            self.mic_combo.clear()
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_combo.addItem(f"{device['name']}", i)
                    
            if self.mic_combo.count() == 0:
                self.mic_combo.addItem("No input devices found", -1)
                
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            self.mic_combo.addItem("Error querying devices", -1)

    def select_audio_file(self):
        """Select an audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Audio File", 
            "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )
        
        if file_path:
            self.current_audio_file = file_path
            filename = Path(file_path).name
            # Truncate long filenames
            if len(filename) > 30:
                filename = filename[:27] + "..."
            self.file_label.setText(f"{filename}")
            self.file_label.setStyleSheet("""
                color: white;
                padding: 6px 15px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            """)
            self.file_radio.setChecked(True)

    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording video and audio"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"jambatalk_output_{timestamp}.mp4"
        
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Recording As",
            output_path,
            "MP4 Video (*.mp4)"
        )
        
        if not output_path:
            return
        
        if not output_path.endswith('.mp4'):
            output_path += '.mp4'
        
        self.recorder = VideoRecorder(output_path, fps=30, sample_rate=SAMPLE_RATE)
        self.recorder.start()
        self.is_recording = True
        
        self.record_button.setText("STOP RECORDING")
        self.record_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EF4444, stop:1 #DC2626);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F87171, stop:1 #EF4444);
            }
            QPushButton:pressed {
                background: #DC2626;
            }
        """)
        self.recording_label.setText("RECORDING IN PROGRESS")
        self.status_label.setText(f"Saving to: {Path(output_path).name}")
        
        print(f"Started recording to: {output_path}")

    def stop_recording(self):
        """Stop recording and save file"""
        if not self.is_recording or not self.recorder:
            return
        
        self.is_recording = False
        self.recording_label.setText("Saving video...")
        self.record_button.setEnabled(False)
        
        def save_thread():
            success = self.recorder.stop()
            if success:
                self.status_label.setText(f"Recording saved: {self.recorder.output_path}")
                print(f"Recording saved successfully!")
            else:
                self.status_label.setText("Failed to save recording")
            
            self.record_button.setText("RECORD")
            self.record_button.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 255, 255, 0.15);
                    border: 2px solid rgba(255, 255, 255, 0.4);
                    border-radius: 12px;
                    padding: 15px;
                    color: white;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.25);
                    border: 2px solid rgba(255, 255, 255, 0.6);
                }
                QPushButton:disabled {
                    background: rgba(255, 255, 255, 0.05);
                    border: 2px solid rgba(255, 255, 255, 0.15);
                    color: rgba(255, 255, 255, 0.4);
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.1);
                }
            """)
            self.record_button.setEnabled(True)
            self.recording_label.setText("")
        
        threading.Thread(target=save_thread, daemon=True).start()

    def toggle_processing(self):
        """Toggle between START/STOP"""
        if not self.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        """Start the processing"""
        try:
            self.clear_queues()
            
            self.running = True
            self.total_frames = 0
            self.last_infer_time = 0.0
            self.window_fps = 0.0
            
            # Start audio playback thread for file mode
            if self.file_radio.isChecked():
                self.playback_thread = AudioPlaybackThread()
                self.playback_thread.start()
            
            # Update button with animation
            self.run_button.setText("STOP")
            self.run_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #EF4444, stop:1 #DC2626);
                    border: none;
                    border-radius: 12px;
                    padding: 15px;
                    color: white;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F87171, stop:1 #EF4444);
                }
                QPushButton:pressed {
                    background: #DC2626;
                }
            """)
            
            # Enable recording button
            self.record_button.setEnabled(True)
            
            # Start appropriate audio input
            if self.mic_radio.isChecked():
                self.start_microphone()
                self.status_label.setText("Processing microphone input...")
            else:
                self.start_file_processing()
                self.status_label.setText("Processing audio file...")
            
            # Start inference thread
            threading.Thread(target=self.inference_thread, daemon=True).start()
            
            # Start display updates
            self.display_timer.start(30)
            
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.running = False

    def start_microphone(self):
        """Start microphone input"""
        device_id = self.mic_combo.currentData()
        if device_id == -1:
            raise Exception("No valid microphone selected")
        
        def recording_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio stream status: {status}")
            chunk = indata[:, 0].copy()
            audio_queue.put(chunk)
            
            if self.is_recording and self.recorder:
                self.recorder.add_audio(chunk)
        
        self.audio_stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=recording_callback
        )
        self.audio_stream.start()
        print("Microphone started")

    def start_file_processing(self):
        """Start audio file processing"""
        if not self.current_audio_file:
            raise Exception("No audio file selected")
            
        if not os.path.exists(self.current_audio_file):
            raise Exception("Audio file not found")

        self.audio_file_processor = AudioFileProcessor(
            self.current_audio_file, 
            audio_queue, 
            1.0
        )
        
        self.audio_file_processor.progress.connect(self.update_progress)
        self.audio_file_processor.finished.connect(self.on_file_finished)
        self.audio_file_processor.error.connect(self.on_file_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.audio_file_processor.start()
        
        # Load and save entire audio for recording
        if self.is_recording and self.recorder:
            try:
                audio, sr = librosa.load(self.current_audio_file, sr=SAMPLE_RATE)
                self.recorder.add_audio(audio)
            except Exception as e:
                print(f"Error loading audio for recording: {e}")
        
        print("File processing started")

    def stop_processing(self):
        """Stop all processing"""
        self.running = False
        
        if self.is_recording:
            self.stop_recording()
        
        self.record_button.setEnabled(False)
        
        # Stop audio playback thread
        if self.playback_thread:
            self.playback_thread.stop()
            self.playback_thread.wait()
            self.playback_thread = None
        
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
            
        if self.audio_file_processor:
            self.audio_file_processor.stop()
            self.audio_file_processor = None
            
        self.display_timer.stop()
        self.progress_bar.setVisible(False)
        
        self.run_button.setText("START")
        self.run_button.setStyleSheet("""
            QPushButton {
                background: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                color: #6366F1;
            }
            QPushButton:hover {
                background: #F8F9FF;
            }
            QPushButton:pressed {
                background: #E8E9FF;
            }
        """)
        
        self.video_label.setText("Click START to begin")
        self.status_label.setText("Ready to begin")
        self.fps_label.setText("FPS: --")
        self.latency_label.setText("Latency: --")
        
        print("Processing stopped")

    def clear_queues(self):
        """Clear all queues"""
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
                
        while not mesh_queue.empty():
            try:
                mesh_queue.get_nowait()
            except queue.Empty:
                break
        
        while not playback_queue.empty():
            try:
                playback_queue.get_nowait()
            except queue.Empty:
                break

    def inference_thread(self):
        """Main inference thread"""
        audio_buffer = np.zeros(0, dtype=np.float32)
        
        print("Running warmup...")
        try:
            warmup_audio = torch.randn(1, 16000).to(self.device).half()
            self.model.predict(warmup_audio, self.template)
            print("Warmup completed!")
        except Exception as e:
            print(f"Warmup failed: {e}")

        print("Inference thread started")
        
        while self.running:
            try:
                chunk = audio_queue.get(timeout=0.5)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Add audio chunk to playback queue for synchronized playback
                if self.playback_thread and self.playback_thread.running:
                    playback_queue.put(chunk)
                
                if len(audio_buffer) >= int(SAMPLE_RATE * WINDOW_DURATION):
                    input_audio = audio_buffer[:int(SAMPLE_RATE * WINDOW_DURATION)]
                    audio_buffer = audio_buffer[int(SAMPLE_RATE * STRIDE_DURATION):]

                    features = extract_features(self.processor, input_audio, self.device)

                    start_time = time.time()
                    prediction, _, _ = self.model.predict(features, self.template)
                    inference_time_sec = time.time() - start_time
                    
                    self.last_infer_time = inference_time_sec * 1000
                    self.window_fps = self.frames_per_window / inference_time_sec if inference_time_sec > 0 else 0

                    if prediction.shape[-1] != self.args.vertice_dim:
                        continue

                    if self.args.dataset == 'vocaset':
                        prediction = prediction.squeeze(0).detach().cpu().numpy().reshape(-1, 5023, 3)
                    elif self.args.dataset == 'BIWI':
                        prediction = prediction.squeeze(0).detach().cpu().numpy().reshape(-1, 3895, 3)

                    prediction = np.nan_to_num(prediction[-1])
                    
                    while not mesh_queue.empty():
                        try:
                            mesh_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    mesh_queue.put_nowait(prediction)
                    self.total_frames += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"Inference error: {e}")

    def update_display(self):
        """Update the video display"""
        if not mesh_queue.empty():
            try:
                mesh = mesh_queue.get_nowait()
                self.renderer.update_mesh(mesh)
                image = self.renderer.render()

                realtime_fps = self.args.fps
                speedup = self.window_fps / realtime_fps if realtime_fps > 0 else 0

                # Update stats labels
                self.fps_label.setText(f"FPS: {self.window_fps:.1f}")
                self.latency_label.setText(f"Latency: {self.last_infer_time:.1f}ms")
                self.frames_label.setText(f"Frames: {self.total_frames}")
                
                # Add minimal performance overlay
                y_offset = 30
                
                # FPS indicator
                # fps_text = f"{self.window_fps:.0f} FPS"
                # cv2.putText(image, fps_text,
                #            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (99, 102, 241), 2)
                
                if self.is_recording:
                    # Recording indicator in top right
                    img_height, img_width = image.shape[:2]
                    rec_text = "REC"
                    (rec_width, rec_height), _ = cv2.getTextSize(
                        rec_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    rec_x = img_width - rec_width - 40
                    rec_y = 35
                    
                    # Pulsing red dot
                    pulse = int((time.time() * 2) % 2)
                    if pulse:
                        cv2.circle(image, (rec_x - 15, rec_y - 8), 6, (220, 38, 38), -1)
                    
                    cv2.putText(image, rec_text, 
                               (rec_x, rec_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 38, 38), 2)

                # Save frame if recording
                if self.is_recording and self.recorder:
                    self.recorder.add_frame(image)

                # Display
                h, w, ch = image.shape
                bytes_per_line = ch * w
                qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                
            except queue.Empty:
                pass

    def update_progress(self, progress):
        """Update file processing progress"""
        self.progress_bar.setValue(progress)

    def on_file_finished(self):
        """Handle file processing completion"""
        self.progress_bar.setValue(100)
        self.status_label.setText("File processing completed")
        print("File processing completed")

    def on_file_error(self, error):
        """Handle file processing errors"""
        self.status_label.setText(f"File error: {error}")
        self.stop_processing()
        print(f"File error: {error}")

    def closeEvent(self, event):
        """Handle window close"""
        if self.is_recording:
            self.stop_recording()
            time.sleep(0.5)
        
        if self.playback_thread:
            self.playback_thread.stop()
            self.playback_thread.wait()
        
        if self.running:
            self.stop_processing()
        event.accept()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="BIWI/biwi.pth")
    parser.add_argument("--template_path", type=str, default="BIWI/templates.pkl")
    parser.add_argument("--subject", type=str, default="F5")
    parser.add_argument("--render_template_path", type=str, default="BIWI/templates_ply/M6.ply")
    parser.add_argument("--dataset", type=str, default="BIWI")
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--vertice_dim", type=int, default=3895 * 3)
    parser.add_argument("--period", type=int, default=25)
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("="*70)
    print("JambaTalk - Real-Time 3D Talking Face with Advanced Recording")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"FPS: {args.fps}")
    print(f"Device: {args.device}")
    print("="*70)
    
    app = QApplication(sys.argv)
    
    # Set modern font with better rendering
    font = QFont("Segoe UI", 10)
    if not font.exactMatch():
        font = QFont("SF Pro Display", 10)
    if not font.exactMatch():
        font = QFont("Ubuntu", 10)
    if not font.exactMatch():
        font = QFont("Arial", 10)
    
    font.setStyleHint(QFont.SansSerif)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    
    window = JambaTalkApp(args)
    window.show()
    sys.exit(app.exec_())
