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
import sounddevice as sd


class SettingsTab(QWidget):
    """Scheda per le impostazioni."""

    settings_changed = pyqtSignal(dict)  # Segnale emesso quando le impostazioni cambiano

    def __init__(self, controller):
        """
        Inizializza la scheda.

        Args:
            controller: Controller dell'applicazione
        """
        super().__init__()

        self.logger = logging.getLogger("ai_parlante.gui.settings_tab")
        self.controller = controller
        self.current_settings = {}

        # Crea l'interfaccia
        self._create_ui()

        # Carica le impostazioni correnti
        self._load_settings()

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
        general_layout.addRow(self.sample_rate_combo)

        # Aggiungi la scheda al tab widget
        settings_tabs.addTab(general_tab, "Generale")

        # Scheda Audio
        audio_tab = QWidget()
        audio_layout = QFormLayout(audio_tab)

        # Dispositivo di input
        audio_layout.addRow(QLabel("Dispositivo di input:"))
        self.input_device_combo = QComboBox()
        self._populate_audio_devices(self.input_device_combo, 'input')
        audio_layout.addRow(self.input_device_combo)

        # Dispositivo di output
        audio_layout.addRow(QLabel("Dispositivo di output:"))
        self.output_device_combo = QComboBox()
        self._populate_audio_devices(self.output_device_combo, 'output')
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
        model_layout.addRow(self.model_size_combo)

        # Numero di epoche predefinito
        model_layout.addRow(QLabel("Epoche predefinite:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        model_layout.addRow(self.epochs_spin)

        # Utilizza CUDA
        self.use_cuda_check = QCheckBox("Utilizza CUDA (se disponibile)")
        model_layout.addRow(self.use_cuda_check)

        # Precisione mista
        self.mixed_precision_check = QCheckBox("Utilizza precisione mista (FP16)")
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
        advanced_layout.addRow(self.fft_size_combo)

        # Hop length
        advanced_layout.addRow(QLabel("Hop length:"))
        self.hop_length_combo = QComboBox()
        self.hop_length_combo.addItems(["128", "256", "512"])
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

        # Pulsante Salva
        self.save_button = QPushButton("Salva Impostazioni")
        buttons_layout.addWidget(self.save_button)

        # Pulsante Applica
        self.apply_button = QPushButton("Applica")
        self.apply_button.setEnabled(False)
        buttons_layout.addWidget(self.apply_button)

        # Aggiungi il layout pulsanti al layout principale
        main_layout.addLayout(buttons_layout)

        # Spaziatore
        main_layout.addStretch()

    @staticmethod
    def _populate_audio_devices(combo_box, device_type):
        """Popola il combobox con i dispositivi audio disponibili."""
        combo_box.clear()
        combo_box.addItem("Dispositivo predefinito", -1)

        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_type == 'input' and device['max_input_channels'] > 0:
                combo_box.addItem(f"{i}: {device['name']}", i)
            elif device_type == 'output' and device['max_output_channels'] > 0:
                combo_box.addItem(f"{i}: {device['name']}", i)

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

        # Pulsante Salva
        self.save_button.clicked.connect(self._on_save_clicked)

        # Pulsante Applica
        self.apply_button.clicked.connect(self._on_apply_clicked)

        # Connessioni per rilevare modifiche
        for widget in self.findChildren((QComboBox, QLineEdit, QSlider, QCheckBox, QSpinBox)):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_setting_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_setting_changed)
            elif isinstance(widget, QSlider):
                widget.valueChanged.connect(self._on_setting_changed)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._on_setting_changed)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._on_setting_changed)

    def _load_settings(self):
        """Carica le impostazioni dal controller."""
        if hasattr(self.controller, 'get_settings'):
            settings = self.controller.get_settings()

            # Imposta i valori nei widget
            self.output_dir_edit.setText(settings.get('output_dir', ''))
            self.models_dir_edit.setText(settings.get('model_dir', ''))
            self.output_format_combo.setCurrentText(settings.get('output_format', 'WAV'))

            sample_rate = settings.get('sample_rate', 22050)
            self.sample_rate_combo.setCurrentText(f"{sample_rate} Hz")

            # Dispositivi audio
            input_device = settings.get('input_device', -1)
            self._select_device(self.input_device_combo, input_device)

            output_device = settings.get('output_device', -1)
            self._select_device(self.output_device_combo, output_device)

            self.volume_slider.setValue(settings.get('volume', 80))
            self.model_size_combo.setCurrentText(settings.get('model_size', 'Medio'))
            self.epochs_spin.setValue(settings.get('epochs', 100))
            self.use_cuda_check.setChecked(settings.get('use_cuda', True))
            self.mixed_precision_check.setChecked(settings.get('mixed_precision', True))
            self.debug_mode_check.setChecked(settings.get('debug_mode', False))
            self.batch_size_spin.setValue(settings.get('batch_size', 8))
            self.fft_size_combo.setCurrentText(str(settings.get('fft_size', 1024)))
            self.hop_length_combo.setCurrentText(str(settings.get('hop_length', 256)))

            self.current_settings = settings.copy()
        else:
            self.logger.warning("Il controller non ha un metodo get_settings()")

    @staticmethod
    def _select_device(combo_box, device_id):
        """Seleziona un dispositivo nel combobox."""
        index = combo_box.findData(device_id)
        if index >= 0:
            combo_box.setCurrentIndex(index)
        else:
            combo_box.setCurrentIndex(0)  # Predefinito

    def _on_setting_changed(self):
        """Gestisce il cambiamento di qualsiasi impostazione."""
        self.apply_button.setEnabled(True)

    def _on_browse_output_clicked(self):
        """Gestisce il click sul pulsante Sfoglia Output."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Directory di Output",
            self.output_dir_edit.text() or os.path.expanduser("~")
        )

        if directory:
            self.output_dir_edit.setText(directory)
            self._on_setting_changed()

    def _on_browse_models_clicked(self):
        """Gestisce il click sul pulsante Sfoglia Modelli."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Directory dei Modelli",
            self.models_dir_edit.text() or os.path.expanduser("~")
        )

        if directory:
            self.models_dir_edit.setText(directory)
            self._on_setting_changed()

    def _on_volume_changed(self, value):
        """Gestisce il cambio di valore dello slider Volume."""
        self.volume_label.setText(f"{value}%")
        self._on_setting_changed()

    def _on_reset_clicked(self):
        """Gestisce il click sul pulsante Ripristina Predefiniti."""
        reply = QMessageBox.question(
            self,
            "Ripristina Predefiniti",
            "Sei sicuro di voler ripristinare tutte le impostazioni ai valori predefiniti?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Ripristina i valori predefiniti
            self.output_dir_edit.setText(os.path.join(os.path.expanduser("~"), "ai_parlante_output"))
            self.models_dir_edit.setText(os.path.join(os.path.expanduser("~"), "ai_parlante_models"))
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

            self.apply_button.setEnabled(True)

    def _on_save_clicked(self):
        """Gestisce il click sul pulsante Salva."""
        self._collect_settings()

        if hasattr(self.controller, 'save_settings'):
            try:
                self.controller.save_settings(self.current_settings)
                QMessageBox.information(
                    self,
                    "Impostazioni Salvate",
                    "Le impostazioni sono state salvate con successo."
                )
                self.apply_button.setEnabled(False)
            except Exception as e:
                self.logger.error(f"Errore durante il salvataggio delle impostazioni: {e}")
                QMessageBox.critical(
                    self,
                    "Errore",
                    f"Si è verificato un errore durante il salvataggio:\n{str(e)}"
                )
        else:
            QMessageBox.warning(
                self,
                "Funzionalità non disponibile",
                "Il controller non supporta il salvataggio diretto delle impostazioni."
            )

    def _on_apply_clicked(self):
        """Gestisce il click sul pulsante Applica."""
        self._collect_settings()

        if hasattr(self.controller, 'apply_settings'):
            try:
                self.controller.apply_settings(self.current_settings)
                QMessageBox.information(
                    self,
                    "Impostazioni Applicate",
                    "Le impostazioni sono state applicate con successo."
                )
                self.apply_button.setEnabled(False)

                # Emetti il segnale per notificare altri componenti
                self.settings_changed.emit(self.current_settings)
            except Exception as e:
                self.logger.error(f"Errore durante l'applicazione delle impostazioni: {e}")
                QMessageBox.critical(
                    self,
                    "Errore",
                    f"Si è verificato un errore durante l'applicazione:\n{str(e)}"
                )
        else:
            QMessageBox.warning(
                self,
                "Funzionalità non disponibile",
                "Il controller non supporta l'applicazione diretta delle impostazioni."
            )

    def _collect_settings(self):
        """Raccoglie tutte le impostazioni dall'interfaccia."""
        self.current_settings = {
            'output_dir': self.output_dir_edit.text(),
            'model_dir': self.models_dir_edit.text(),
            'output_format': self.output_format_combo.currentText(),
            'sample_rate': int(self.sample_rate_combo.currentText().split()[0]),
            'input_device': self.input_device_combo.currentData(),
            'output_device': self.output_device_combo.currentData(),
            'volume': self.volume_slider.value(),
            'model_size': self.model_size_combo.currentText(),
            'epochs': self.epochs_spin.value(),
            'use_cuda': self.use_cuda_check.isChecked(),
            'mixed_precision': self.mixed_precision_check.isChecked(),
            'debug_mode': self.debug_mode_check.isChecked(),
            'batch_size': self.batch_size_spin.value(),
            'fft_size': int(self.fft_size_combo.currentText()),
            'hop_length': int(self.hop_length_combo.currentText())
        }