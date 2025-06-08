#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Parlante con Riferimento Vocale
Script principale che avvia l'applicazione.
"""

import warnings
import os
import sys
import argparse
import logging
from datetime import datetime

warnings.filterwarnings("ignore")

# Configurazione del logging
def setup_logging():
    """Configura il sistema di logging."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"ai_parlante_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("ai_parlante")


# Verifica delle dipendenze
def check_dependencies():
    """Verifica che tutte le dipendenze necessarie siano disponibili."""
    try:
        import torch
        import numpy
        import librosa
        import PyQt5

        # Verifica CUDA
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logging.warning("CUDA non disponibile. L'applicazione funzionerà in modalità CPU, ma sarà molto più lenta.")
        else:
            device_name = torch.cuda.get_device_name(0)
            logging.info(f"Dispositivo CUDA rilevato: {device_name}")

        return True
    except ImportError as e:
        logging.error(f"Dipendenza mancante: {e}")
        return False


def parse_arguments():
    """Analizza gli argomenti della riga di comando."""
    parser = argparse.ArgumentParser(description="AI Parlante con Riferimento Vocale")

    parser.add_argument("--no-gui", action="store_true", help="Esegui in modalità console senza interfaccia grafica")
    parser.add_argument("--input-audio", type=str, help="File audio di riferimento")
    parser.add_argument("--output-dir", type=str, help="Directory di output per i file generati")
    parser.add_argument("--model-dir", type=str, help="Directory per i modelli vocali")
    parser.add_argument("--debug", action="store_true", help="Attiva la modalità debug")

    return parser.parse_args()


def main():
    """Funzione principale dell'applicazione."""
    # Setup logging
    logger = setup_logging()
    logger.info("Avvio dell'applicazione AI Parlante")

    # Analisi argomenti
    args = parse_arguments()

    # Imposta il livello di log in base alla modalità debug
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modalità debug attivata")

    # Verifica dipendenze
    if not check_dependencies():
        logger.error("Verifica delle dipendenze fallita. Uscita.")
        return 1

    # Importa i moduli necessari
    try:
        # Importa il controller solo dopo aver verificato le dipendenze
        from controller import Controller

        if args.no_gui:
            # Modalità console
            logger.info("Avvio in modalità console")
            from console_app import ConsoleApp
            app = ConsoleApp(args)
            return app.run()
        else:
            # Modalità GUI
            logger.info("Avvio dell'interfaccia grafica")
            from PyQt5.QtWidgets import QApplication
            from gui.main_window import MainWindow

            # Crea l'applicazione Qt
            qt_app = QApplication(sys.argv)

            # Crea il controller
            controller = Controller(
                input_audio=args.input_audio,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                debug=args.debug
            )

            # Crea e mostra la finestra principale
            main_window = MainWindow(controller)
            main_window.show()

            # Esegui il loop dell'applicazione
            return qt_app.exec_()

    except Exception as e:
        logger.exception(f"Errore durante l'avvio dell'applicazione: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

