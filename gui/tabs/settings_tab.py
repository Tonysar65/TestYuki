"""
Scheda per le impostazioni.
Permette all'utente di configurare vari parametri dell'applicazione.
"""

import logging

import sounddevice as sd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QComboBox, QSlider, QCheckBox, QSpinBox, QTabWidget, QFormLayout
)


class SettingsTab(QWidget):
    """Scheda per le impostazioni."""

    settings_changed = pyqtSignal(dict)  # Segnale emesso quando le impostazioni cambiano

    def __init__(self, controller):
        """Inizializza la scheda."""
        super().__init__()
        self.logger = logging.getLogger("YukiAI.gui.settings_tab")
        self.controller = controller
        self.current_settings = {}

        self._create_ui()
        self._load_settings()
        self._connect_signals()

        self.logger.info("Scheda impostazioni inizializzata")

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        settings_tabs = QTabWidget()

        # Scheda Generale
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)

        # Directory di output
        general_layout.addRow(QLabel("Directory di output:"))
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.browse_output_button = QPushButton("Sfoglia...")
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.browse_output_button)
        general_layout.addRow(output_layout)

        # Directory dei modelli
        general_layout.addRow(QLabel("Directory dei modelli:"))
        models_layout = QHBoxLayout()
        self.models_dir_edit = QLineEdit()
        self.models_dir_edit.setReadOnly(True)
        self.browse_models_button = QPushButton("Sfoglia...")
        models_layout.addWidget(self.models_dir_edit)
        models_layout.addWidget(self.browse_models_button)
        general_layout.addRow(models_layout)

        # Formato di output
        general_layout.addRow(QLabel("Formato di output predefinito:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["WAV", "MP3", "FLAC", "OGG"])
        general_layout.addRow(self.output_format_combo)

        # Frequenza di campionamento
        general_layout.addRow(QLabel("Frequenza di campionamento:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["8000 Hz", "16000 Hz", "22050 Hz", "44100 Hz", "48000 Hz"])
        general_layout.addRow(self.sample_rate_combo)

        settings_tabs.addTab(general_tab, "Generale")

        # Scheda Audio
        audio_tab = QWidget()
        audio_layout = QFormLayout(audio_tab)

        # Dispositivi audio
        audio_layout.addRow(QLabel("Dispositivo di input:"))
        self.input_device_combo = QComboBox()
        self._populate_audio_devices(self.input_device_combo, 'input')
        audio_layout.addRow(self.input_device_combo)

        audio_layout.addRow(QLabel("Dispositivo di output:"))
        self.output_device_combo = QComboBox()
        self._populate_audio_devices(self.output_device_combo, 'output')
        audio_layout.addRow(self.output_device_combo)

        # Volume
        audio_layout.addRow(QLabel("Volume di riproduzione:"))
        volume_layout = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_label = QLabel("80%")
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_label)
        audio_layout.addRow(volume_layout)

        settings_tabs.addTab(audio_tab, "Audio")

        # Scheda Modello
        model_tab = QWidget()
        model_layout = QFormLayout(model_tab)

        # Parametri del modello
        model_layout.addRow(QLabel("Dimensione del modello:"))
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["Piccolo", "Medio", "Grande"])
        model_layout.addRow(self.model_size_combo)

        model_layout.addRow(QLabel("Epoche predefinite:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        model_layout.addRow(self.epochs_spin)

        # Opzioni GPU
        self.use_cuda_check = QCheckBox("Utilizza CUDA (se disponibile)")
        model_layout.addRow(self.use_cuda_check)

        self.mixed_precision_check = QCheckBox("Utilizza precisione mista (FP16)")
        model_layout.addRow(self.mixed_precision_check)

        settings_tabs.addTab(model_tab, "Modello")

        # Scheda Avanzate
        advanced_tab = QWidget()
        advanced_layout = QFormLayout(advanced_tab)

        # Debug
        self.debug_mode_check = QCheckBox("Modalità debug")
        advanced_layout.addRow(self.debug_mode_check)

        # Parametri avanzati
        advanced_layout.addRow(QLabel("Dimensione del batch:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        advanced_layout.addRow(self.batch_size_spin)

        advanced_layout.addRow(QLabel("Dimensione FFT:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["512", "1024", "2048", "4096"])
        advanced_layout.addRow(self.fft_size_combo)

        advanced_layout.addRow(QLabel("Hop length:"))
        self.hop_length_combo = QComboBox()
        self.hop_length_combo.addItems(["128", "256", "512"])
        advanced_layout.addRow(self.hop_length_combo)

        settings_tabs.addTab(advanced_tab, "Avanzate")
        main_layout.addWidget(settings_tabs)

        # Pulsanti
        buttons_layout = QHBoxLayout()
        self.reset_button = QPushButton("Ripristina Predefiniti")
        self.save_button = QPushButton("Salva Impostazioni")
        self.apply_button = QPushButton("Applica")
        self.apply_button.setEnabled(False)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.apply_button)
        main_layout.addLayout(buttons_layout)

    def _populate_audio_devices(self, combo_box, device_type):
        """Popola il combobox con i dispositivi audio disponibili."""
        combo_box.clear()
        combo_box.addItem("Dispositivo predefinito", -1)

        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device_type == 'input' and device['max_input_channels'] > 0:
                    combo_box.addItem(f"{i}: {device['name']}", i)
                elif device_type == 'output' and device['max_output_channels'] > 0:
                    combo_box.addItem(f"{i}: {device['name']}", i)
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento dei dispositivi audio: {e}")

    def _select_device(self, combo_box, device_index):
        """Seleziona un dispositivo nel combobox."""
        for i in range(combo_box.count()):
            if combo_box.itemData(i) == device_index:
                combo_box.setCurrentIndex(i)
                break

    def _connect_signals(self):
        """Connette i segnali."""
        # Pulsanti
        self.browse_output_button.clicked.connect(self._on_browse_output_clicked)
        self.browse_models_button.clicked.connect(self._on_browse_models_clicked)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.apply_button.clicked.connect(self._on_apply_clicked)

        # Controlli
        self.volume_slider.valueChanged.connect(self._on_volume_changed)

        # Monitoraggio cambiamenti
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

    def _on_browse_output_clicked(self):
        """Gestisce il click sul pulsante per selezionare la directory di output."""
        directory = QFileDialog.getExistingDirectory(self, "Seleziona directory di output")
        if directory:
            self.output_dir_edit.setText(directory)
            self._on_setting_changed()

    def _on_browse_models_clicked(self):
        """Gestisce il click sul pulsante per selezionare la directory dei modelli."""
        directory = QFileDialog.getExistingDirectory(self, "Seleziona directory dei modelli")
        if directory:
            self.models_dir_edit.setText(directory)
            self._on_setting_changed()

    def _on_volume_changed(self, value):
        """Aggiorna l'etichetta del volume quando cambia il valore dello slider."""
        self.volume_label.setText(f"{value}%")
        self._on_setting_changed()

    def _on_setting_changed(self):
        """Abilita il pulsante Applica quando cambia un'impostazione."""
        self.apply_button.setEnabled(True)

    def _on_save_clicked(self):
        """Salva le impostazioni."""
        self._apply_settings()
        if hasattr(self.controller, 'save_settings'):
            try:
                self.controller.save_settings(self.current_settings)
                QMessageBox.information(self, "Salvataggio", "Impostazioni salvate correttamente.")
                self.apply_button.setEnabled(False)
            except Exception as e:
                self.logger.error(f"Errore durante il salvataggio delle impostazioni: {e}")
                QMessageBox.critical(self, "Errore", f"Impossibile salvare le impostazioni: {str(e)}")

    def _on_apply_clicked(self):
        """Applica le impostazioni senza salvarle permanentemente."""
        self._apply_settings()
        self.settings_changed.emit(self.current_settings)
        self.apply_button.setEnabled(False)

    def _apply_settings(self):
        """Aggrega tutte le impostazioni in un dizionario."""
        sample_rate = int(self.sample_rate_combo.currentText().split()[0])

        self.current_settings = {
            'output_dir': self.output_dir_edit.text(),
            'model_dir': self.models_dir_edit.text(),
            'output_format': self.output_format_combo.currentText(),
            'sample_rate': sample_rate,
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
            'hop_length': int(self.hop_length_combo.currentText()),
        }

    def _load_settings(self):
        """Carica le impostazioni esistenti."""
        if hasattr(self.controller, 'get_settings'):
            try:
                settings = self.controller.get_settings()

                # Impostazioni generali
                self.output_dir_edit.setText(settings.get('output_dir', ''))
                self.models_dir_edit.setText(settings.get('model_dir', ''))
                self.output_format_combo.setCurrentText(settings.get('output_format', 'WAV'))
                self.sample_rate_combo.setCurrentText(f"{settings.get('sample_rate', 22050)} Hz")

                # Impostazioni audio
                self._select_device(self.input_device_combo, settings.get('input_device', -1))
                self._select_device(self.output_device_combo, settings.get('output_device', -1))
                self.volume_slider.setValue(settings.get('volume', 80))

                # Impostazioni modello
                self.model_size_combo.setCurrentText(settings.get('model_size', 'Medio'))
                self.epochs_spin.setValue(settings.get('epochs', 100))
                self.use_cuda_check.setChecked(settings.get('use_cuda', True))
                self.mixed_precision_check.setChecked(settings.get('mixed_precision', True))

                # Impostazioni avanzate
                self.debug_mode_check.setChecked(settings.get('debug_mode', False))
                self.batch_size_spin.setValue(settings.get('batch_size', 8))
                self.fft_size_combo.setCurrentText(str(settings.get('fft_size', 1024)))
                self.hop_length_combo.setCurrentText(str(settings.get('hop_length', 256)))

            except Exception as e:
                self.logger.error(f"Errore durante il caricamento delle impostazioni: {e}")
                QMessageBox.warning(self, "Errore", f"Impossibile caricare le impostazioni: {str(e)}")

    def _on_reset_clicked(self):
        """Ripristina le impostazioni predefinite."""
        reply = QMessageBox.question(
            self, "Conferma",
            "Sei sicuro di voler ripristinare le impostazioni predefinite?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                if hasattr(self.controller, 'reset_to_defaults'):
                    self.controller.reset_to_defaults()
                self._load_settings()
                self.apply_button.setEnabled(True)
            except Exception as e:
                self.logger.error(f"Errore durante il ripristino delle impostazioni: {e}")
                QMessageBox.critical(self, "Errore", f"Impossibile ripristinare le impostazioni: {str(e)}")