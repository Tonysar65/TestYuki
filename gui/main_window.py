#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Finestra principale dell'applicazione AI Parlante.
"""

import os
import sys
import logging
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QStatusBar, QFileDialog, QMessageBox,
    QApplication, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QFont

# Importa le schede
from gui.tabs.voice_cloning_tab import VoiceCloningTab
from gui.tabs.synthesis_tab import SynthesisTab
from gui.tabs.settings_tab import SettingsTab


class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione."""

    def __init__(self, controller):
        """
        Inizializza la finestra principale.

        Args:
            controller: Controller dell'applicazione
        """
        super().__init__()

        self.logger = logging.getLogger("ai_parlante.gui.main_window")
        self.controller = controller

        # Imposta il titolo e le dimensioni
        self.setWindowTitle("AI Parlante - Sintesi Vocale con Riferimento Audio")
        self.resize(1000, 700)

        # Crea l'interfaccia
        self._create_ui()

        # Connetti i segnali
        self._connect_signals()

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

        # Logo (placeholder)
        logo_label = QLabel()
        logo_label.setFixedSize(64, 64)
        logo_label.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        header_layout.addWidget(logo_label)

        # Titolo
        title_label = QLabel("AI Parlante")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        header_layout.addWidget(title_label)

        # Spaziatore
        header_layout.addStretch()

        # Aggiungi l'intestazione al layout principale
        main_layout.addLayout(header_layout)

        # Separatore
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Tab widget
        self.tab_widget = QTabWidget()

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
        self.setStatusBar(self.status_bar)

        # Etichetta di stato
        self.status_label = QLabel("Pronto")
        self.status_bar.addWidget(self.status_label, 1)

        # Barra di progresso (placeholder)
        self.progress_label = QLabel("0%")
        self.status_bar.addPermanentWidget(self.progress_label)

    def _connect_signals(self):
        """Connette i segnali."""
        # Connetti il segnale di cambio stato del controller
        self.controller.on_state_changed = self._on_state_changed
        self.controller.on_progress_changed = self._on_progress_changed
        self.controller.on_status_changed = self._on_status_changed

    def _on_state_changed(self, state):
        """
        Gestisce il cambio di stato del controller.

        Args:
            state: Nuovo stato
        """
        # Aggiorna l'interfaccia in base allo stato
        pass

    def _on_progress_changed(self, progress):
        """
        Gestisce il cambio di progresso del controller.

        Args:
            progress: Nuovo valore del progresso (0.0-1.0)
        """
        # Aggiorna la barra di progresso
        progress_percent = int(progress * 100)
        self.progress_label.setText(f"{progress_percent}%")

    def _on_status_changed(self, message):
        """
        Gestisce il cambio di messaggio di stato del controller.

        Args:
            message: Nuovo messaggio di stato
        """
        # Aggiorna l'etichetta di stato
        self.status_label.setText(message)

    def closeEvent(self, event):
        """
        Gestisce l'evento di chiusura della finestra.

        Args:
            event: Evento di chiusura
        """
        # Esegui la pulizia delle risorse
        self.controller.cleanup()

        # Accetta l'evento di chiusura
        event.accept()

