"""
Scheda per la sintesi vocale.
Permette all'utente di selezionare un modello vocale addestrato e sintetizzare il parlato a partire dal testo.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSlider, QCheckBox, QSpinBox,
    QTextEdit, QToolButton
)
from PyQt5.QtCore import Qt, QSize, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QFont


class SynthesisTab(QWidget):
    """Scheda per la sintesi vocale."""

    # Definiamo i segnali personalizzati
    synthesis_started = pyqtSignal()
    synthesis_progress = pyqtSignal(int)
    synthesis_completed = pyqtSignal()

    def __init__(self, controller):
        """
        Inizializza la scheda.

        Args:
            controller: Controller dell'applicazione
        """
        super().__init__()

        self.logger = logging.getLogger("ai_parlante.gui.synthesis_tab")
        self.controller = controller

        # Crea l'interfaccia
        self._create_ui()

        # Connetti i segnali
        self._connect_signals()

        # Carica i modelli disponibili
        self._load_models()

        self.logger.info("Scheda sintesi vocale inizializzata")

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        # Layout principale
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Gruppo Modello Vocale
        model_group = QGroupBox("Modello Vocale")
        model_layout = QHBoxLayout(model_group)

        # Selezione del modello
        model_layout.addWidget(QLabel("Modello:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo, 1)

        # Pulsante Ricarica
        self.reload_button = QPushButton("Ricarica")
        model_layout.addWidget(self.reload_button)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(model_group)

        # Gruppo Testo da Sintetizzare
        text_group = QGroupBox("Testo da Sintetizzare")
        text_layout = QVBoxLayout(text_group)

        # Editor di testo
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Inserisci il testo da sintetizzare...")
        text_layout.addWidget(self.text_edit)

        # Layout pulsanti
        buttons_layout = QHBoxLayout()

        # Pulsante Pulisci
        self.clear_button = QPushButton("Pulisci")
        buttons_layout.addWidget(self.clear_button)

        # Pulsante Carica da File
        self.load_text_button = QPushButton("Carica da File")
        buttons_layout.addWidget(self.load_text_button)

        # Aggiungi il layout pulsanti al layout del gruppo
        text_layout.addLayout(buttons_layout)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(text_group)

        # Gruppo Sintesi
        synthesis_group = QGroupBox("Sintesi Vocale")
        synthesis_layout = QGridLayout(synthesis_group)

        # Parametri di sintesi
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

        # Pulsante Sintetizza
        self.synthesize_button = QPushButton("Sintetizza")
        synthesis_layout.addWidget(self.synthesize_button, 2, 0, 1, 3)

        # Barra di progresso
        self.synthesis_progress = QProgressBar()
        self.synthesis_progress.setRange(0, 100)
        self.synthesis_progress.setValue(0)
        synthesis_layout.addWidget(self.synthesis_progress, 3, 0, 1, 3)

        # Stato della sintesi
        self.synthesis_status = QLabel("In attesa")
        synthesis_layout.addWidget(self.synthesis_status, 4, 0, 1, 3)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(synthesis_group)

        # Gruppo Riproduzione
        playback_group = QGroupBox("Riproduzione")
        playback_layout = QHBoxLayout(playback_group)

        # Pulsante Riproduci
        self.play_button = QPushButton("Riproduci")
        self.play_button.setEnabled(False)
        playback_layout.addWidget(self.play_button)

        # Pulsante Pausa
        self.pause_button = QPushButton("Pausa")
        self.pause_button.setEnabled(False)
        playback_layout.addWidget(self.pause_button)

        # Pulsante Stop
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        playback_layout.addWidget(self.stop_button)

        # Pulsante Salva
        self.save_button = QPushButton("Salva Audio")
        self.save_button.setEnabled(False)
        playback_layout.addWidget(self.save_button)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(playback_group)

        # Spaziatore
        main_layout.addStretch()

    def _connect_signals(self):
        """Connette i segnali."""
        # Pulsante Ricarica
        self.reload_button.clicked.connect(self._load_models)

        # Pulsante Pulisci
        self.clear_button.clicked.connect(self.text_edit.clear)

        # Pulsante Carica da File
        self.load_text_button.clicked.connect(self._on_load_text_clicked)

        # Slider Velocità
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        # Slider Tono
        self.pitch_slider.valueChanged.connect(self._on_pitch_changed)

        # Pulsante Sintetizza
        self.synthesize_button.clicked.connect(self._on_synthesize_clicked)

        # Pulsante Riproduci
        self.play_button.clicked.connect(self._on_play_clicked)

        # Pulsante Pausa
        self.pause_button.clicked.connect(self._on_pause_clicked)

        # Pulsante Stop
        self.stop_button.clicked.connect(self._on_stop_clicked)

        # Pulsante Salva
        self.save_button.clicked.connect(self._on_save_clicked)

        # Connessione dei segnali personalizzati
        self.synthesis_progress.valueChanged.connect(self.synthesis_progress.setValue)
        self.synthesis_completed.connect(self._on_synthesis_completed)

    def _load_models(self):
        """Carica i modelli disponibili."""
        # Salva il modello selezionato corrente
        current_model = self.model_combo.currentText()

        # Pulisci la combo box
        self.model_combo.clear()

        # Ottieni i modelli disponibili
        models = self.controller.get_available_models()

        if models:
            # Aggiungi i modelli alla combo box
            self.model_combo.addItems(models)

            # Ripristina il modello selezionato
            if current_model in models:
                index = self.model_combo.findText(current_model)
                self.model_combo.setCurrentIndex(index)

            # Abilita il pulsante Sintetizza
            self.synthesize_button.setEnabled(True)
        else:
            # Nessun modello disponibile
            self.model_combo.addItem("Nessun modello disponibile")

            # Disabilita il pulsante Sintetizza
            self.synthesize_button.setEnabled(False)

    def _on_load_text_clicked(self):
        """Gestisce il click sul pulsante Carica da File."""
        # Apri il dialogo di selezione file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona File di Testo",
            "",
            "File di Testo (*.txt);;Tutti i file (*.*)"
        )

        if file_path:
            try:
                # Leggi il file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Imposta il testo nell'editor
                self.text_edit.setText(text)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Errore",
                    f"Impossibile leggere il file: {str(e)}"
                )

    def _on_speed_changed(self, value):
        """
        Gestisce il cambio di valore dello slider Velocità.

        Args:
            value: Nuovo valore
        """
        # Aggiorna l'etichetta
        self.speed_label.setText(f"{value}%")

    def _on_pitch_changed(self, value):
        """
        Gestisce il cambio di valore dello slider Tono.

        Args:
            value: Nuovo valore
        """
        # Aggiorna l'etichetta
        self.pitch_label.setText(f"{value:+d}")

    def _on_synthesize_clicked(self):
        """Gestisce il click sul pulsante Sintetizza."""
        # Ottieni il modello selezionato
        model_name = self.model_combo.currentText()

        if model_name == "Nessun modello disponibile":
            QMessageBox.warning(
                self,
                "Nessun Modello",
                "Nessun modello vocale disponibile. Addestra un modello nella scheda Clonazione Vocale."
            )
            return

        # Ottieni il testo
        text = self.text_edit.toPlainText()

        if not text:
            QMessageBox.warning(
                self,
                "Testo Mancante",
                "Inserisci il testo da sintetizzare."
            )
            return

        # Aggiorna lo stato
        self.synthesis_status.setText("Sintesi in corso...")

        # Disabilita i pulsanti
        self.synthesize_button.setEnabled(False)

        # Avvia la sintesi
        self.synthesis_started.emit()
        self._simulate_synthesis()

    def _on_play_clicked(self):
        """Gestisce il click sul pulsante Riproduci."""
        # Aggiorna lo stato
        self.synthesis_status.setText("Riproduzione in corso...")

        # Abilita/disabilita i pulsanti
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def _on_pause_clicked(self):
        """Gestisce il click sul pulsante Pausa."""
        # Aggiorna lo stato
        self.synthesis_status.setText("Riproduzione in pausa")

        # Abilita/disabilita i pulsanti
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _on_stop_clicked(self):
        """Gestisce il click sul pulsante Stop."""
        # Aggiorna lo stato
        self.synthesis_status.setText("Riproduzione interrotta")

        # Abilita/disabilita i pulsanti
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _on_save_clicked(self):
        """Gestisce il click sul pulsante Salva."""
        # Apri il dialogo di salvataggio file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salva File Audio",
            "",
            "File WAV (*.wav);;File MP3 (*.mp3)"
        )

        if file_path:
            # Aggiorna lo stato
            self.synthesis_status.setText(f"Audio salvato: {os.path.basename(file_path)}")

            # Mostra un messaggio
            QMessageBox.information(
                self,
                "Audio Salvato",
                f"Il file audio è stato salvato come {file_path}"
            )

    def _simulate_synthesis(self):
        """Simula la sintesi vocale."""
        # Crea un timer per simulare il progresso
        self.synthesis_timer = QTimer(self)
        self.synthesis_timer.timeout.connect(self._update_synthesis_progress)
        self.synthesis_progress.setValue(0)
        self.synthesis_timer.start(50)

    def _update_synthesis_progress(self):
        """Aggiorna il progresso della sintesi."""
        current_value = self.synthesis_progress.value()
        if current_value < 100:
            self.synthesis_progress.setValue(current_value + 1)
            self.synthesis_progress.emit(current_value + 1)
        else:
            # Ferma il timer
            self.synthesis_timer.stop()

            # Completa la sintesi
            self.synthesis_completed.emit()

    def _on_synthesis_completed(self):
        """Gestisce il completamento della sintesi."""
        # Aggiorna lo stato
        self.synthesis_status.setText("Sintesi completata")

        # Abilita i pulsanti
        self.synthesize_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)