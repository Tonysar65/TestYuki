#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scheda per le impostazioni.
Permette all'utente di configurare vari parametri dell'applicazione.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QComboBox, QSlider, QCheckBox, QSpinBox,
    QTabWidget, QFormLayout
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QFont


class SettingsTab(QWidget):
    """Scheda per le impostazioni."""

    def __init__(self, controller):
        """
        Inizializza la scheda.

        Args:
            controller: Controller dell'applicazione
        """
        super().__init__()

        self.logger = logging.getLogger("ai_parlante.gui.settings_tab")
        self.controller = controller

        # Crea l'interfaccia
        self._create_ui()

        # Connetti i segnali
        self._connect_signals()

        self.logger.info("Scheda impostazioni inizializzata")

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        # Layout principale
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget per le impostazioni
        settings_tabs = QTabWidget()

        # Scheda Generale
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)

        # Directory di output
        general_layout.addRow(QLabel("Directory di output:"))

        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        output_layout.addWidget(self.output_dir_edit)

        self.browse_output_button = QPushButton("Sfoglia...")
        output_layout.addWidget(self.browse_output_button)

        general_layout.addRow(output_layout)

        # Directory dei modelli
        general_layout.addRow(QLabel("Directory dei modelli:"))

        models_layout = QHBoxLayout()
        self.models_dir_edit = QLineEdit()
        self.models_dir_edit.setReadOnly(True)
        models_layout.addWidget(self.models_dir_edit)

        self.browse_models_button = QPushButton("Sfoglia...")
        models_layout.addWidget(self.browse_models_button)

        general_layout.addRow(models_layout)

        # Formato di output predefinito
        general_layout.addRow(QLabel("Formato di output predefinito:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["WAV", "MP3", "FLAC", "OGG"])
        general_layout.addRow(self.output_format_combo)

        # Frequenza di campionamento predefinita
        general_layout.addRow(QLabel("Frequenza di campionamento:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["8000 Hz", "16000 Hz", "22050 Hz", "44100 Hz", "48000 Hz"])
        self.sample_rate_combo.setCurrentText("22050 Hz")
        general_layout.addRow(self.sample_rate_combo)

        # Aggiungi la scheda al tab widget
        settings_tabs.addTab(general_tab, "Generale")

        # Scheda Audio
        audio_tab = QWidget()
        audio_layout = QFormLayout(audio_tab)

        # Dispositivo di input
        audio_layout.addRow(QLabel("Dispositivo di input:"))
        self.input_device_combo = QComboBox()
        self.input_device_combo.addItem("Dispositivo predefinito")
        audio_layout.addRow(self.input_device_combo)

        # Dispositivo di output
        audio_layout.addRow(QLabel("Dispositivo di output:"))
        self.output_device_combo = QComboBox()
        self.output_device_combo.addItem("Dispositivo predefinito")
        audio_layout.addRow(self.output_device_combo)

        # Volume di riproduzione
        audio_layout.addRow(QLabel("Volume di riproduzione:"))

        volume_layout = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        volume_layout.addWidget(self.volume_slider)

        self.volume_label = QLabel("80%")
        volume_layout.addWidget(self.volume_label)

        audio_layout.addRow(volume_layout)

        # Aggiungi la scheda al tab widget
        settings_tabs.addTab(audio_tab, "Audio")

        # Scheda Modello
        model_tab = QWidget()
        model_layout = QFormLayout(model_tab)

        # Dimensione del modello
        model_layout.addRow(QLabel("Dimensione del modello:"))
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["Piccolo", "Medio", "Grande"])
        self.model_size_combo.setCurrentText("Medio")
        model_layout.addRow(self.model_size_combo)

        # Numero di epoche predefinito
        model_layout.addRow(QLabel("Epoche predefinite:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        model_layout.addRow(self.epochs_spin)

        # Utilizza CUDA
        self.use_cuda_check = QCheckBox("Utilizza CUDA (se disponibile)")
        self.use_cuda_check.setChecked(True)
        model_layout.addRow(self.use_cuda_check)

        # Precisione mista
        self.mixed_precision_check = QCheckBox("Utilizza precisione mista (FP16)")
        self.mixed_precision_check.setChecked(True)
        model_layout.addRow(self.mixed_precision_check)

        # Aggiungi la scheda al tab widget
        settings_tabs.addTab(model_tab, "Modello")

        # Scheda Avanzate
        advanced_tab = QWidget()
        advanced_layout = QFormLayout(advanced_tab)

        # Modalità debug
        self.debug_mode_check = QCheckBox("Modalità debug")
        advanced_layout.addRow(self.debug_mode_check)

        # Dimensione del batch
        advanced_layout.addRow(QLabel("Dimensione del batch:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        advanced_layout.addRow(self.batch_size_spin)

        # Dimensione della FFT
        advanced_layout.addRow(QLabel("Dimensione FFT:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["512", "1024", "2048", "4096"])
        self.fft_size_combo.setCurrentText("1024")
        advanced_layout.addRow(self.fft_size_combo)

        # Hop length
        advanced_layout.addRow(QLabel("Hop length:"))
        self.hop_length_combo = QComboBox()
        self.hop_length_combo.addItems(["128", "256", "512"])
        self.hop_length_combo.setCurrentText("256")
        advanced_layout.addRow(self.hop_length_combo)

        # Aggiungi la scheda al tab widget
        settings_tabs.addTab(advanced_tab, "Avanzate")

        # Aggiungi il tab widget al layout principale
        main_layout.addWidget(settings_tabs)

        # Pulsanti
        buttons_layout = QHBoxLayout()

        # Pulsante Ripristina Predefiniti
        self.reset_button = QPushButton("Ripristina Predefiniti")
        buttons_layout.addWidget(self.reset_button)

        # Spaziatore
        buttons_layout.addStretch()

        # Pulsante Applica
        self.apply_button = QPushButton("Applica")
        buttons_layout.addWidget(self.apply_button)

        # Aggiungi il layout pulsanti al layout principale
        main_layout.addLayout(buttons_layout)

        # Spaziatore
        main_layout.addStretch()

    def _connect_signals(self):
        """Connette i segnali."""
        # Pulsante Sfoglia Output
        self.browse_output_button.clicked.connect(self._on_browse_output_clicked)

        # Pulsante Sfoglia Modelli
        self.browse_models_button.clicked.connect(self._on_browse_models_clicked)

        # Slider Volume
        self.volume_slider.valueChanged.connect(self._on_volume_changed)

        # Pulsante Ripristina Predefiniti
        self.reset_button.clicked.connect(self._on_reset_clicked)

        # Pulsante Applica
        self.apply_button.clicked.connect(self._on_apply_clicked)

    def _on_browse_output_clicked(self):
        """Gestisce il click sul pulsante Sfoglia Output."""
        # Apri il dialogo di selezione directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Directory di Output",
            ""
        )

        if directory:
            # Imposta la directory
            self.output_dir_edit.setText(directory)

    def _on_browse_models_clicked(self):
        """Gestisce il click sul pulsante Sfoglia Modelli."""
        # Apri il dialogo di selezione directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Directory dei Modelli",
            ""
        )

        if directory:
            # Imposta la directory
            self.models_dir_edit.setText(directory)

    def _on_volume_changed(self, value):
        """
        Gestisce il cambio di valore dello slider Volume.

        Args:
            value: Nuovo valore
        """
        # Aggiorna l'etichetta
        self.volume_label.setText(f"{value}%")

    def _on_reset_clicked(self):
        """Gestisce il click sul pulsante Ripristina Predefiniti."""
        # Chiedi conferma
        reply = QMessageBox.question(
            self,
            "Ripristina Predefiniti",
            "Sei sicuro di voler ripristinare tutte le impostazioni ai valori predefiniti?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Ripristina i valori predefiniti
            self.output_format_combo.setCurrentText("WAV")
            self.sample_rate_combo.setCurrentText("22050 Hz")
            self.input_device_combo.setCurrentIndex(0)
            self.output_device_combo.setCurrentIndex(0)
            self.volume_slider.setValue(80)
            self.model_size_combo.setCurrentText("Medio")
            self.epochs_spin.setValue(100)
            self.use_cuda_check.setChecked(True)
            self.mixed_precision_check.setChecked(True)
            self.debug_mode_check.setChecked(False)
            self.batch_size_spin.setValue(8)
            self.fft_size_combo.setCurrentText("1024")
            self.hop_length_combo.setCurrentText("256")

            # Mostra un messaggio
            QMessageBox.information(
                self,
                "Impostazioni Ripristinate",
                "Le impostazioni sono state ripristinate ai valori predefiniti."
            )

    def _on_apply_clicked(self):
        """Gestisce il click sul pulsante Applica."""
        # Salva le impostazioni
        # In un'implementazione reale, si utilizzerebbero le impostazioni del controller

        # Mostra un messaggio
        QMessageBox.information(
            self,
            "Impostazioni Applicate",
            "Le impostazioni sono state applicate con successo."
        )

