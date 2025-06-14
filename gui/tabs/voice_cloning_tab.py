"""
Scheda per la clonazione vocale.
Permette all'utente di caricare un file audio di riferimento e addestrare un modello vocale.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSlider, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QFont

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


class VoiceCloningTab(QWidget):
    """Scheda per la clonazione vocale."""

    def __init__(self, controller):
        """
        Inizializza la scheda.

        Args:
            controller: Controller dell'applicazione
        """
        super().__init__()

        self.logger = logging.getLogger("ai_parlante.gui.voice_cloning_tab")
        self.controller = controller

        # Crea l'interfaccia
        self._create_ui()

        # Connetti i segnali
        self._connect_signals()

        self.logger.info("Scheda clonazione vocale inizializzata")

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        # Layout principale
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Gruppo File Audio di Riferimento
        reference_group = QGroupBox("File Audio di Riferimento")
        reference_layout = QGridLayout(reference_group)

        # Percorso del file
        reference_layout.addWidget(QLabel("File:"), 0, 0)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        reference_layout.addWidget(self.file_path_edit, 0, 1)

        # Pulsante Sfoglia
        self.browse_button = QPushButton("Sfoglia...")
        reference_layout.addWidget(self.browse_button, 0, 2)

        # Informazioni sul file
        reference_layout.addWidget(QLabel("Durata:"), 1, 0)
        self.duration_label = QLabel("--:--")
        reference_layout.addWidget(self.duration_label, 1, 1)

        reference_layout.addWidget(QLabel("Campionamento:"), 2, 0)
        self.sample_rate_label = QLabel("-- Hz")
        reference_layout.addWidget(self.sample_rate_label, 2, 1)

        # Pulsante Riproduci
        self.play_button = QPushButton("Riproduci")
        self.play_button.setEnabled(False)
        reference_layout.addWidget(self.play_button, 1, 2)

        # Pulsante Stop
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        reference_layout.addWidget(self.stop_button, 2, 2)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(reference_group)

        # Gruppo Analisi Audio
        analysis_group = QGroupBox("Analisi Audio")
        analysis_layout = QVBoxLayout(analysis_group)

        # Pulsante Analizza
        self.analyze_button = QPushButton("Analizza Audio")
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)

        # Barra di progresso
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        analysis_layout.addWidget(self.analysis_progress)

        # Stato dell'analisi
        self.analysis_status = QLabel("In attesa del file audio")
        analysis_layout.addWidget(self.analysis_status)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(analysis_group)

        # Gruppo Addestramento Modello
        training_group = QGroupBox("Addestramento Modello")
        training_layout = QGridLayout(training_group)

        # Nome del modello
        training_layout.addWidget(QLabel("Nome del modello:"), 0, 0)
        self.model_name_edit = QLineEdit()
        training_layout.addWidget(self.model_name_edit, 0, 1, 1, 2)

        # Parametri di addestramento
        training_layout.addWidget(QLabel("Epoche:"), 1, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        training_layout.addWidget(self.epochs_spin, 1, 1)

        training_layout.addWidget(QLabel("Qualità:"), 2, 0)
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 5)
        self.quality_slider.setValue(3)
        training_layout.addWidget(self.quality_slider, 2, 1)
        self.quality_label = QLabel("Media")
        training_layout.addWidget(self.quality_label, 2, 2)

        # Pulsante Addestra
        self.train_button = QPushButton("Addestra Modello")
        self.train_button.setEnabled(False)
        training_layout.addWidget(self.train_button, 3, 0, 1, 3)

        # Barra di progresso
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        training_layout.addWidget(self.training_progress, 4, 0, 1, 3)

        # Stato dell'addestramento
        self.training_status = QLabel("In attesa dell'analisi audio")
        training_layout.addWidget(self.training_status, 5, 0, 1, 3)

        # Aggiungi il gruppo al layout principale
        main_layout.addWidget(training_group)

        # Spaziatore
        main_layout.addStretch()

    def _connect_signals(self):
        """Connette i segnali."""
        # Pulsante Sfoglia
        self.browse_button.clicked.connect(self._on_browse_clicked)

        # Pulsante Riproduci
        self.play_button.clicked.connect(self._on_play_clicked)

        # Pulsante Stop
        self.stop_button.clicked.connect(self._on_stop_clicked)

        # Pulsante Analizza
        self.analyze_button.clicked.connect(self._on_analyze_clicked)

        # Slider Qualità
        self.quality_slider.valueChanged.connect(self._on_quality_changed)

        # Pulsante Addestra
        self.train_button.clicked.connect(self._on_train_clicked)

    def _on_browse_clicked(self):
        """Gestisce il click sul pulsante Sfoglia."""
        # Apri il dialogo di selezione file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona File Audio",
            "",
            "File Audio (*.wav *.mp3 *.flac *.ogg);;Tutti i file (*.*)"
        )

        if file_path:
            # Imposta il percorso del file
            self.file_path_edit.setText(file_path)

            # Abilita i pulsanti
            self.play_button.setEnabled(True)
            self.analyze_button.setEnabled(True)

            # Aggiorna lo stato
            self.analysis_status.setText("File audio selezionato")

            # Carica il file audio
            self.controller.load_reference_audio(file_path)

    def _on_play_clicked(self):
        """Gestisce il click sul pulsante Riproduci."""
        # Riproduci il file audio
        file_path = self.file_path_edit.text()
        if file_path and os.path.exists(file_path):
            self.controller.play_audio(file_path)

            # Abilita il pulsante Stop
            self.stop_button.setEnabled(True)

    def _on_stop_clicked(self):
        """Gestisce il click sul pulsante Stop."""
        # Interrompi la riproduzione
        self.controller.stop_playback()

        # Disabilita il pulsante Stop
        self.stop_button.setEnabled(False)

    def _on_analyze_clicked(self):
        """Gestisce il click sul pulsante Analizza."""
        # Analizza il file audio
        file_path = self.file_path_edit.text()
        if file_path and os.path.exists(file_path):
            # Aggiorna lo stato
            self.analysis_status.setText("Analisi in corso...")

            # Disabilita i pulsanti
            self.analyze_button.setEnabled(False)

            # Simula l'analisi (in un'implementazione reale, si utilizzerebbe il controller)
            self._simulate_analysis()

    def _on_quality_changed(self, value):
        """
        Gestisce il cambio di valore dello slider Qualità.

        Args:
            value: Nuovo valore
        """
        # Aggiorna l'etichetta
        quality_labels = ["Molto bassa", "Bassa", "Media", "Alta", "Molto alta"]
        self.quality_label.setText(quality_labels[value - 1])

    def _on_train_clicked(self):
        """Gestisce il click sul pulsante Addestra."""
        model_name = self.model_name_edit.text().strip()
        epochs = self.epochs_spin.value()
        quality = self.quality_slider.value()

        if not model_name:
            QMessageBox.warning(
                self,
                "Nome Modello Mancante",
                "Inserisci un nome per il modello."
            )
            return

        self.training_status.setText("Addestramento in corso...")
        self.train_button.setEnabled(False)

        try:
            # Chiamata reale al controller
            self.controller.train_model(model_name, epochs, quality)

            self.training_status.setText("Addestramento completato")
            QMessageBox.information(
                self,
                "Modello Addestrato",
                f"Il modello '{model_name}' è stato salvato in 'models/'."
            )
        except Exception as e:
            self.training_status.setText("Errore durante l'addestramento")
            QMessageBox.critical(
                self,
                "Errore",
                f"Si è verificato un errore:\n{str(e)}"
            )
        finally:
            self.train_button.setEnabled(True)

    def _simulate_analysis(self):
        """Simula l'analisi audio."""
        # Crea un timer per simulare il progresso
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self._update_analysis_progress)
        self.analysis_progress.setValue(0)
        self.analysis_timer.start(100)

    def _update_analysis_progress(self):
        """Aggiorna il progresso dell'analisi."""
        current_value = self.analysis_progress.value()
        if current_value < 100:
            self.analysis_progress.setValue(current_value + 1)
        else:
            # Ferma il timer
            self.analysis_timer.stop()

            # Aggiorna lo stato
            self.analysis_status.setText("Analisi completata")

            # Abilita i pulsanti
            self.analyze_button.setEnabled(True)
            self.train_button.setEnabled(True)

    def _simulate_training(self):
        """Esegue l'addestramento del modello."""
        # Carica i dati di addestramento (esempio con dati casuali)
        X = np.random.rand(100, 10)  # 100 campioni, 10 caratteristiche
        y = np.random.randint(0, 2, 100)  # 100 etichette binarie

        # Suddividi i dati in set di addestramento e di test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crea e addestra il modello
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # Salva il modello addestrato
        self._save_model(model)

        # Calcola e stampa l'accuratezza sul set di test
        accuracy = model.score(X_test, y_test)
        print(f"Accuratezza del modello: {accuracy * 100:.2f}%")

    def _update_training_progress(self):
        """Aggiorna il progresso dell'addestramento."""
        current_value = self.training_progress.value()
        if current_value < 100:
            self.training_progress.setValue(current_value + 1)
        else:
            # Ferma il timer
            self.training_timer.stop()

            # Aggiorna lo stato
            self.training_status.setText("Addestramento completato")

            # Abilita i pulsanti
            self.train_button.setEnabled(True)

            # Mostra un messaggio
            QMessageBox.information(
                self,
                "Addestramento Completato",
                f"Il modello '{self.model_name_edit.text()}' è stato addestrato con successo."
            )