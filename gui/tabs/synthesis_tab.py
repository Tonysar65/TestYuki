"""
Scheda per la sintesi vocale.
Permette all'utente di selezionare un modello vocale addestrato e sintetizzare il parlato a partire dal testo.
"""

import logging
import os

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSlider, QTextEdit
)


class SynthesisTab(QWidget):
    """Scheda per la sintesi vocale."""

    synthesis_started = pyqtSignal()
    synthesis_progress = pyqtSignal(int)
    synthesis_completed = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.logger = logging.getLogger("YukiAI.gui.synthesis_tab")
        self.controller = controller
        self.current_audio = None
        self.is_playing = False

        self._create_ui()
        self._connect_signals()
        self._load_models()

        self.logger.info("Scheda sintesi vocale inizializzata")

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Gruppo Modello Vocale
        model_group = QGroupBox("Modello Vocale")
        model_layout = QHBoxLayout(model_group)

        model_layout.addWidget(QLabel("Modello:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo, 1)

        self.reload_button = QPushButton("Ricarica")
        model_layout.addWidget(self.reload_button)

        main_layout.addWidget(model_group)

        # Gruppo Testo da Sintetizzare
        text_group = QGroupBox("Testo da Sintetizzare")
        text_layout = QVBoxLayout(text_group)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Inserisci il testo da sintetizzare...")
        text_layout.addWidget(self.text_edit)

        buttons_layout = QHBoxLayout()
        self.clear_button = QPushButton("Pulisci")
        self.load_text_button = QPushButton("Carica da File")
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.load_text_button)
        text_layout.addLayout(buttons_layout)

        main_layout.addWidget(text_group)

        # Gruppo Sintesi
        synthesis_group = QGroupBox("Sintesi Vocale")
        synthesis_layout = QGridLayout(synthesis_group)

        synthesis_layout.addWidget(QLabel("Velocità:"), 0, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        synthesis_layout.addWidget(self.speed_slider, 0, 1)
        self.speed_label = QLabel("100%")
        synthesis_layout.addWidget(self.speed_label, 0, 2)

        synthesis_layout.addWidget(QLabel("Tono:"), 1, 0)
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.pitch_slider.setValue(0)
        synthesis_layout.addWidget(self.pitch_slider, 1, 1)
        self.pitch_label = QLabel("0")
        synthesis_layout.addWidget(self.pitch_label, 1, 2)

        self.synthesize_button = QPushButton("Sintetizza")
        synthesis_layout.addWidget(self.synthesize_button, 2, 0, 1, 3)

        self.synthesis_progress = QProgressBar()
        self.synthesis_progress.setRange(0, 100)
        self.synthesis_progress.setValue(0)
        synthesis_layout.addWidget(self.synthesis_progress, 3, 0, 1, 3)

        self.synthesis_status = QLabel("In attesa")
        synthesis_layout.addWidget(self.synthesis_status, 4, 0, 1, 3)

        main_layout.addWidget(synthesis_group)

        # Gruppo Riproduzione
        playback_group = QGroupBox("Riproduzione")
        playback_layout = QHBoxLayout(playback_group)

        self.play_button = QPushButton("Riproduci")
        self.play_button.setEnabled(False)
        self.pause_button = QPushButton("Pausa")
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Salva Audio")
        self.save_button.setEnabled(False)

        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.pause_button)
        playback_layout.addWidget(self.stop_button)
        playback_layout.addWidget(self.save_button)

        main_layout.addWidget(playback_group)
        main_layout.addStretch()

    def _connect_signals(self):
        """Connette i segnali."""
        self.reload_button.clicked.connect(self._load_models)
        self.clear_button.clicked.connect(self.text_edit.clear)
        self.load_text_button.clicked.connect(self._on_load_text_clicked)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.pitch_slider.valueChanged.connect(self._on_pitch_changed)
        self.synthesize_button.clicked.connect(self._on_synthesize_clicked)
        self.play_button.clicked.connect(self._on_play_clicked)
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)

        self.synthesis_progress.valueChanged.connect(self.synthesis_progress.setValue)
        self.synthesis_completed.connect(self._on_synthesis_completed)

    def _load_models(self):
        """Carica i modelli disponibili."""
        current_model = self.model_combo.currentText()
        self.model_combo.clear()

        try:
            models = self.controller.get_available_models()
            if models:
                self.model_combo.addItems(models)
                if current_model in models:
                    self.model_combo.setCurrentIndex(self.model_combo.findText(current_model))
                self.synthesize_button.setEnabled(True)
            else:
                self.model_combo.addItem("Nessun modello disponibile")
                self.synthesize_button.setEnabled(False)
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dei modelli: {e}")
            QMessageBox.warning(self, "Errore", f"Impossibile caricare i modelli: {str(e)}")

    def _on_load_text_clicked(self):
        """Carica testo da file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona File di Testo", "",
            "File di Testo (*.txt);;Tutti i file (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.text_edit.setText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "Errore", f"Impossibile leggere il file: {str(e)}")

    def _on_speed_changed(self, value):
        """Aggiorna la velocità di riproduzione."""
        self.speed_label.setText(f"{value}%")

    def _on_pitch_changed(self, value):
        """Aggiorna il tono di riproduzione."""
        self.pitch_label.setText(f"{value:+d}")

    def _on_synthesize_clicked(self):
        """Avvia la sintesi vocale."""
        model_name = self.model_combo.currentText()
        text = self.text_edit.toPlainText()

        if not model_name or model_name == "Nessun modello disponibile":
            QMessageBox.warning(self, "Errore", "Seleziona un modello valido")
            return

        if not text.strip():
            QMessageBox.warning(self, "Errore", "Inserisci del testo da sintetizzare")
            return

        try:
            self.synthesis_status.setText("Sintesi in corso...")
            self.synthesize_button.setEnabled(False)

            # Parametri di sintesi
            speed = self.speed_slider.value() / 100.0
            pitch = self.pitch_slider.value()

            # Avvia la sintesi (simulata)
            self.synthesis_started.emit()
            self._simulate_synthesis()

        except Exception as e:
            self.logger.error(f"Errore nella sintesi: {e}")
            QMessageBox.critical(self, "Errore", f"Errore durante la sintesi: {str(e)}")
            self.synthesize_button.setEnabled(True)

    def _on_play_clicked(self):
        """Avvia la riproduzione dell'audio."""
        if self.current_audio:
            self.is_playing = True
            self.synthesis_status.setText("Riproduzione in corso...")
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)

    def _on_pause_clicked(self):
        """Mette in pausa la riproduzione."""
        self.is_playing = False
        self.synthesis_status.setText("Riproduzione in pausa")
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def _on_stop_clicked(self):
        """Interrompe la riproduzione."""
        self.is_playing = False
        self.synthesis_status.setText("Riproduzione interrotta")
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _on_save_clicked(self):
        """Salva l'audio generato."""
        if not self.current_audio:
            QMessageBox.warning(self, "Errore", "Nessun audio da salvare")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salva File Audio", "",
            "File WAV (*.wav);;File MP3 (*.mp3)"
        )

        if file_path:
            try:
                # Qui andrebbe il codice per salvare l'audio
                self.synthesis_status.setText(f"Audio salvato: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Successo", f"File salvato: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Impossibile salvare il file: {str(e)}")

    def _simulate_synthesis(self):
        """Simula il progresso della sintesi."""
        self.synthesis_timer = QTimer(self)
        self.synthesis_timer.timeout.connect(self._update_synthesis_progress)
        self.synthesis_progress.setValue(0)
        self.synthesis_timer.start(50)

    def _update_synthesis_progress(self):
        """Aggiorna la barra di progresso."""
        current_value = self.synthesis_progress.value()
        if current_value < 100:
            self.synthesis_progress.setValue(current_value + 1)
        else:
            self.synthesis_timer.stop()
            self._on_synthesis_completed()

    def _on_synthesis_completed(self):
        """Gestisce il completamento della sintesi."""
        self.current_audio = "audio_generato"  # Sostituire con l'audio reale
        self.synthesis_status.setText("Sintesi completata")
        self.synthesize_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.synthesis_completed.emit()