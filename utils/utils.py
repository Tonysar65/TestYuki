#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilità per l'interfaccia utente grafica.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSlider, QCheckBox, QSpinBox,
    QTabWidget, QFormLayout, QDialog, QFrame
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QIcon, QPixmap, QFont, QPalette, QColor


class WaveformWidget(QWidget):
    """Widget per la visualizzazione della forma d'onda audio."""

    def __init__(self, parent=None):
        """
        Inizializza il widget.

        Args:
            parent: Widget genitore
        """
        super().__init__(parent)

        self.logger = logging.getLogger("ai_parlante.gui.waveform_widget")

        # Dati
        self.waveform = None
        self.sample_rate = None

        # Colori
        self.background_color = QColor(240, 240, 240)
        self.waveform_color = QColor(0, 120, 215)
        self.grid_color = QColor(200, 200, 200)

        # Inizializza l'interfaccia
        self._create_ui()

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        # Imposta le dimensioni minime
        self.setMinimumSize(300, 100)

        # Imposta lo sfondo
        palette = self.palette()
        palette.setColor(QPalette.Window, self.background_color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def set_waveform(self, waveform, sample_rate):
        """
        Imposta la forma d'onda da visualizzare.

        Args:
            waveform: Forma d'onda audio
            sample_rate: Frequenza di campionamento
        """
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.update()

    def paintEvent(self, event):
        """
        Gestisce l'evento di disegno.

        Args:
            event: Evento di disegno
        """
        import numpy as np
        from PyQt5.QtGui import QPainter, QPen

        # Crea il painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Disegna lo sfondo
        painter.fillRect(event.rect(), self.background_color)

        # Disegna la griglia
        pen = QPen(self.grid_color)
        pen.setWidth(1)
        painter.setPen(pen)

        # Linee orizzontali
        h_step = self.height() / 4
        for i in range(1, 4):
            y = i * h_step
            painter.drawLine(0, y, self.width(), y)

        # Linee verticali
        v_step = self.width() / 10
        for i in range(1, 10):
            x = i * v_step
            painter.drawLine(x, 0, x, self.height())

        # Disegna la forma d'onda
        if self.waveform is not None and len(self.waveform) > 0:
            # Imposta la penna
            pen = QPen(self.waveform_color)
            pen.setWidth(1)
            painter.setPen(pen)

            # Calcola i punti
            num_points = min(self.width() * 2, len(self.waveform))
            step = len(self.waveform) / num_points

            points = []
            for i in range(num_points):
                idx = int(i * step)
                x = i * self.width() / num_points
                y = (1 - (self.waveform[idx] + 1) / 2) * self.height()
                points.append((x, y))

            # Disegna la linea
            for i in range(1, len(points)):
                painter.drawLine(
                    points[i - 1][0], points[i - 1][1],
                    points[i][0], points[i][1]
                )

        # Termina il painter
        painter.end()


class SpectrogramWidget(QWidget):
    """Widget per la visualizzazione dello spettrogramma audio."""

    def __init__(self, parent=None):
        """
        Inizializza il widget.

        Args:
            parent: Widget genitore
        """
        super().__init__(parent)

        self.logger = logging.getLogger("ai_parlante.gui.spectrogram_widget")

        # Dati
        self.spectrogram = None
        self.sample_rate = None

        # Colori
        self.background_color = QColor(240, 240, 240)

        # Inizializza l'interfaccia
        self._create_ui()

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        # Imposta le dimensioni minime
        self.setMinimumSize(300, 100)

        # Imposta lo sfondo
        palette = self.palette()
        palette.setColor(QPalette.Window, self.background_color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def set_spectrogram(self, spectrogram, sample_rate):
        """
        Imposta lo spettrogramma da visualizzare.

        Args:
            spectrogram: Spettrogramma audio
            sample_rate: Frequenza di campionamento
        """
        self.spectrogram = spectrogram
        self.sample_rate = sample_rate
        self.update()

    def paintEvent(self, event):
        """
        Gestisce l'evento di disegno.

        Args:
            event: Evento di disegno
        """
        import numpy as np
        from PyQt5.QtGui import QPainter, QImage

        # Crea il painter
        painter = QPainter(self)

        # Disegna lo sfondo
        painter.fillRect(event.rect(), self.background_color)

        # Disegna lo spettrogramma
        if self.spectrogram is not None:
            # Normalizza lo spettrogramma
            spec_norm = (self.spectrogram - self.spectrogram.min()) / (
                        self.spectrogram.max() - self.spectrogram.min() + 1e-8)

            # Converti in immagine
            spec_img = (spec_norm * 255).astype(np.uint8)

            # Ridimensiona l'immagine
            h, w = spec_img.shape
            scale_h = self.height() / h
            scale_w = self.width() / w

            # Crea l'immagine Qt
            img = QImage(spec_img.data, w, h, w, QImage.Format_Grayscale8)

            # Disegna l'immagine
            painter.drawImage(event.rect(), img)

        # Termina il painter
        painter.end()


class AudioRecorder(QObject):
    """Classe per la registrazione audio."""

    # Segnali
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(object, int)  # waveform, sample_rate
    recording_error = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        Inizializza il registratore.

        Args:
            parent: Oggetto genitore
        """
        super().__init__(parent)

        self.logger = logging.getLogger("ai_parlante.gui.audio_recorder")

        # Stato
        self.is_recording = False

        # Thread di registrazione
        self.recording_thread = None

        # Dati
        self.waveform = None
        self.sample_rate = 22050

    def start_recording(self):
        """Avvia la registrazione."""
        if self.is_recording:
            return

        try:
            import pyaudio
            import numpy as np
            import threading

            # Imposta lo stato
            self.is_recording = True
            self.waveform = np.array([])

            # Parametri di registrazione
            self.sample_rate = 22050
            chunk_size = 1024
            format = pyaudio.paFloat32
            channels = 1

            # Inizializza PyAudio
            self.audio = pyaudio.PyAudio()

            # Funzione di callback
            def callback(in_data, frame_count, time_info, status):
                if self.is_recording:
                    # Converti i dati in numpy array
                    data = np.frombuffer(in_data, dtype=np.float32)

                    # Aggiungi i dati alla forma d'onda
                    self.waveform = np.append(self.waveform, data)

                    return (in_data, pyaudio.paContinue)
                else:
                    return (in_data, pyaudio.paComplete)

            # Apri lo stream
            self.stream = self.audio.open(
                format=format,
                channels=channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                stream_callback=callback
            )

            # Avvia lo stream
            self.stream.start_stream()

            # Emetti il segnale
            self.recording_started.emit()

            self.logger.info("Registrazione avviata")

        except Exception as e:
            self.logger.error(f"Errore durante l'avvio della registrazione: {e}")
            self.recording_error.emit(str(e))
            self.is_recording = False

    def stop_recording(self):
        """Interrompe la registrazione."""
        if not self.is_recording:
            return

        try:
            # Imposta lo stato
            self.is_recording = False

            # Ferma lo stream
            if hasattr(self, "stream") and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()

            # Chiudi PyAudio
            if hasattr(self, "audio"):
                self.audio.terminate()

            # Emetti il segnale
            if self.waveform is not None and len(self.waveform) > 0:
                self.recording_stopped.emit(self.waveform, self.sample_rate)

            self.logger.info("Registrazione interrotta")

        except Exception as e:
            self.logger.error(f"Errore durante l'interruzione della registrazione: {e}")
            self.recording_error.emit(str(e))


class StyleHelper:
    """Classe per la gestione degli stili dell'interfaccia."""

    @staticmethod
    def apply_dark_theme(app):
        """
        Applica il tema scuro all'applicazione.

        Args:
            app: Applicazione Qt
        """
        # Imposta la tavolozza dei colori
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        # Applica la tavolozza
        app.setPalette(palette)

        # Imposta il foglio di stile
        app.setStyleSheet("""
            QToolTip { 
                color: #ffffff; 
                background-color: #2a82da; 
                border: 1px solid white; 
            }

            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 1ex;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }

            QPushButton {
                background-color: #2a82da;
                color: white;
                border: 1px solid #2a82da;
                padding: 5px;
                border-radius: 3px;
            }

            QPushButton:hover {
                background-color: #3a92ea;
                border: 1px solid #3a92ea;
            }

            QPushButton:pressed {
                background-color: #1a72ca;
                border: 1px solid #1a72ca;
            }

            QPushButton:disabled {
                background-color: #555555;
                border: 1px solid #555555;
                color: #aaaaaa;
            }

            QLineEdit, QTextEdit, QComboBox {
                background-color: #333333;
                color: white;
                border: 1px solid gray;
                padding: 2px;
                border-radius: 3px;
            }

            QProgressBar {
                border: 1px solid gray;
                border-radius: 3px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #2a82da;
                width: 10px;
                margin: 0.5px;
            }

            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #333333;
                margin: 2px 0;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #2a82da;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: #3a92ea;
            }

            QTabWidget::pane {
                border: 1px solid gray;
                border-radius: 3px;
            }

            QTabBar::tab {
                background-color: #333333;
                color: white;
                padding: 5px;
                border: 1px solid gray;
                border-bottom-color: #333333;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }

            QTabBar::tab:selected {
                background-color: #2a82da;
                border-bottom-color: #2a82da;
            }

            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)

    @staticmethod
    def apply_light_theme(app):
        """
        Applica il tema chiaro all'applicazione.

        Args:
            app: Applicazione Qt
        """
        # Reimposta la tavolozza predefinita
        app.setPalette(QPalette())

        # Imposta il foglio di stile
        app.setStyleSheet("""
            QToolTip { 
                color: #000000; 
                background-color: #ffffff; 
                border: 1px solid black; 
            }

            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 1ex;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }

            QPushButton {
                background-color: #0078d7;
                color: white;
                border: 1px solid #0078d7;
                padding: 5px;
                border-radius: 3px;
            }

            QPushButton:hover {
                background-color: #1088e7;
                border: 1px solid #1088e7;
            }

            QPushButton:pressed {
                background-color: #0068c7;
                border: 1px solid #0068c7;
            }

            QPushButton:disabled {
                background-color: #cccccc;
                border: 1px solid #cccccc;
                color: #666666;
            }

            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid gray;
                padding: 2px;
                border-radius: 3px;
            }

            QProgressBar {
                border: 1px solid gray;
                border-radius: 3px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #0078d7;
                width: 10px;
                margin: 0.5px;
            }

            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: #1088e7;
            }

            QTabWidget::pane {
                border: 1px solid gray;
                border-radius: 3px;
            }

            QTabBar::tab {
                background-color: #f0f0f0;
                color: black;
                padding: 5px;
                border: 1px solid gray;
                border-bottom-color: #f0f0f0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }

            QTabBar::tab:selected {
                background-color: #0078d7;
                color: white;
                border-bottom-color: #0078d7;
            }

            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)