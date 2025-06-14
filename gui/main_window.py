"""
Finestra principale dell'applicazione AI Parlante.
"""

import logging
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QStatusBar, QMessageBox,
    QFrame, QProgressBar
)

from gui.tabs.settings_tab import SettingsTab
from gui.tabs.synthesis_tab import SynthesisTab
# Importa le schede
from gui.tabs.voice_cloning_tab import VoiceCloningTab


class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione."""

    def __init__(self, controller):
        """
        Inizializza la finestra principale.

        Args:
            controller: Controller dell'applicazione
        """
        super().__init__()

        self.logger = logging.getLogger("YukiAI.gui.main_window")
        self.controller = controller
        self.current_state = "ready"  # Stati possibili: ready, processing, error

        # Imposta il titolo e le dimensioni
        self.setWindowTitle("AI Parlante - Sintesi Vocale con Riferimento Audio")
        self.resize(1000, 700)

        # Crea l'interfaccia
        self._create_ui()

        # Connetti i segnali
        self._connect_signals()

        # Carica le impostazioni iniziali
        self._load_initial_settings()

        self.logger.info("Finestra principale inizializzata")

    def _create_ui(self):
        """Crea l'interfaccia utente."""
        # Widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principale
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Intestazione
        header_layout = QHBoxLayout()

        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap(os.path.join("assets", "logo.png"))
        if logo_pixmap.isNull():
            logo_label.setFixedSize(64, 64)
            logo_label.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
            logo_label.setText("Logo")
        else:
            logo_label.setPixmap(logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio))
        header_layout.addWidget(logo_label)

        # Titolo
        title_label = QLabel("AI Parlante")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title_label)

        # Spaziatore
        header_layout.addStretch()

        # Pulsante informazioni
        self.info_button = QPushButton()
        self.info_button.setIcon(QIcon.fromTheme("help-about"))
        self.info_button.setToolTip("Informazioni sull'applicazione")
        self.info_button.setFixedSize(32, 32)
        self.info_button.setStyleSheet("QPushButton { border-radius: 16px; }")
        header_layout.addWidget(self.info_button)

        # Aggiungi l'intestazione al layout principale
        main_layout.addLayout(header_layout)

        # Separatore
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(separator)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                padding: 8px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
        """)

        # Scheda Clonazione Vocale
        self.voice_cloning_tab = VoiceCloningTab(self.controller)
        self.tab_widget.addTab(self.voice_cloning_tab, "Clonazione Vocale")

        # Scheda Sintesi Vocale
        self.synthesis_tab = SynthesisTab(self.controller)
        self.tab_widget.addTab(self.synthesis_tab, "Sintesi Vocale")

        # Scheda Impostazioni
        self.settings_tab = SettingsTab(self.controller)
        self.tab_widget.addTab(self.settings_tab, "Impostazioni")

        # Aggiungi il tab widget al layout principale
        main_layout.addWidget(self.tab_widget)

        # Barra di stato
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #ecf0f1;")
        self.setStatusBar(self.status_bar)

        # Etichetta di stato
        self.status_label = QLabel("Pronto")
        self.status_label.setStyleSheet("color: #2c3e50;")
        self.status_bar.addWidget(self.status_label, 1)

        # Barra di progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setTextVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Etichetta di percentuale
        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet("color: #2c3e50; min-width: 40px;")
        self.status_bar.addPermanentWidget(self.progress_label)

        # Pulsante di stop
        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon.fromTheme("process-stop"))
        self.stop_button.setToolTip("Interrompi operazione corrente")
        self.stop_button.setFixedSize(24, 24)
        self.stop_button.setEnabled(False)
        self.status_bar.addPermanentWidget(self.stop_button)

    def _connect_signals(self):
        """Connette i segnali."""
        # Connetti i segnali del controller
        self.controller.on_state_changed = self._on_state_changed
        self.controller.on_progress_changed = self._on_progress_changed
        self.controller.on_status_changed = self._on_status_changed

        # Connetti i pulsanti
        self.info_button.clicked.connect(self._show_about_dialog)
        self.stop_button.clicked.connect(self.controller.cancel_processing)

        # Connetti il segnale delle impostazioni modificate
        self.settings_tab.settings_changed.connect(self._on_settings_changed)

    def _load_initial_settings(self):
        """Carica le impostazioni iniziali."""
        if hasattr(self.controller, 'get_settings'):
            settings = self.controller.get_settings()
            self._apply_ui_settings(settings)

    def _apply_ui_settings(self, settings):
        """Applica le impostazioni all'interfaccia."""
        # Qui puoi applicare le impostazioni che influenzano l'interfaccia
        if 'theme' in settings:
            self._apply_theme(settings['theme'])

    def _apply_theme(self, theme_name):
        """Applica un tema all'interfaccia."""
        # Implementa la logica per cambiare tema
        pass

    def _on_state_changed(self, state):
        """
        Gestisce il cambio di stato del controller.

        Args:
            state: Nuovo stato (ready, processing, error)
        """
        self.current_state = state
        self.logger.debug(f"Cambio stato: {state}")

        # Aggiorna l'interfaccia in base allo stato
        if state == "processing":
            self._set_processing_ui()
        elif state == "error":
            self._set_error_ui()
        else:  # ready
            self._set_ready_ui()

    def _set_ready_ui(self):
        """Configura l'interfaccia per lo stato 'pronto'."""
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")

        # Abilita tutte le schede
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, True)

    def _set_processing_ui(self):
        """Configura l'interfaccia per lo stato 'elaborazione'."""
        self.stop_button.setEnabled(True)

        # Disabilita le schede non correnti durante l'elaborazione
        current_index = self.tab_widget.currentIndex()
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, i == current_index)

    def _set_error_ui(self):
        """Configura l'interfaccia per lo stato 'errore'."""
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("ERR")

        # Ripristina l'abilitazione delle schede
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, True)

    def _on_progress_changed(self, progress):
        """
        Gestisce il cambio di progresso del controller.

        Args:
            progress: Nuovo valore del progresso (0.0-1.0)
        """
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)
        self.progress_label.setText(f"{progress_percent}%")

    def _on_status_changed(self, message):
        """
        Gestisce il cambio di messaggio di stato del controller.

        Args:
            message: Nuovo messaggio di stato
        """
        self.status_label.setText(message)

    def _on_settings_changed(self, settings):
        """
        Gestisce il cambiamento delle impostazioni.

        Args:
            settings: Dizionario con le nuove impostazioni
        """
        self.logger.debug("Impostazioni modificate")
        self._apply_ui_settings(settings)

    def _show_about_dialog(self):
        """Mostra la finestra di dialogo 'Informazioni'."""
        about_text = """
        <h2>AI Parlante</h2>
        <p>Versione 1.0.0</p>
        <p>Applicazione per la sintesi vocale con riferimento audio.</p>
        <p>Sviluppato da Your Company &copy; 2023</p>
        <p>Licenza: GPL v3</p>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Informazioni su AI Parlante")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(about_text)

        # Aggiungi il logo se disponibile
        logo_pixmap = QPixmap(os.path.join("assets", "logo.png"))
        if not logo_pixmap.isNull():
            msg_box.setIconPixmap(logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio))

        msg_box.exec_()

    def closeEvent(self, event):
        """
        Gestisce l'evento di chiusura della finestra.

        Args:
            event: Evento di chiusura
        """
        if self.current_state == "processing":
            reply = QMessageBox.question(
                self,
                "Operazione in corso",
                "Un'operazione è ancora in corso. Vuoi davvero uscire?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

        # Esegui la pulizia delle risorse
        self.controller.cleanup()

        # Salva le impostazioni prima di chiudere
        if hasattr(self.controller, 'save_settings'):
            try:
                self.controller.save_settings()
            except Exception as e:
                self.logger.error(f"Errore durante il salvataggio delle impostazioni: {e}")

        # Accetta l'evento di chiusura
        event.accept()