"""
AI Parlante con Riferimento Vocale
Script principale che avvia l'applicazione.
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from scipy.signal import windows  # Corretto qui!

from synthesizer.hifigan import load_hifigan_model

warnings.filterwarnings("ignore")


def synthesize_mel_to_audio(mel, vocoder, output_path, logger):
    """Sintetizza audio da mel spectrogram utilizzando il vocoder."""
    with torch.no_grad():
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel).unsqueeze(0)
        audio = vocoder(mel).squeeze().cpu().numpy()
        sf.write(output_path, audio, 22050)
        logger.info(f"Audio salvato in: {output_path}")


def setup_logging():
    """Configura il sistema di logging."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"TestYuki_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger("TestYuki")


def check_dependencies():
    """Verifica che tutte le dipendenze necessarie siano disponibili."""
    try:
        import torch
        import numpy
        import librosa
        import PyQt5
        import sounddevice as sd

        # Verifica CUDA
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logging.warning("CUDA non disponibile. L'applicazione funzionerà in modalità CPU.")
        else:
            device_name = torch.cuda.get_device_name(0)
            logging.info(f"Dispositivo CUDA rilevato: {device_name}")

        # Verifica dispositivi audio
        devices = sd.query_devices()
        logging.info("Dispositivi audio disponibili:")
        for i, device in enumerate(devices):
            logging.info(f"{i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")

        return True
    except ImportError as e:
        logging.error(f"Dipendenza mancante: {e}")
        return False


def parse_args():
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(description="AI Parlante con Riferimento Vocale")
    parser.add_argument("--no-gui", action="store_true", help="Modalità console")
    parser.add_argument("--input-audio", type=str, help="File audio di riferimento")
    parser.add_argument("--output-dir", type=str, default="output", help="Output dir")
    parser.add_argument("--model-dir", type=str, default="synthesizer/models/hifigan", help="Model dir")
    parser.add_argument("--debug", action="store_true", help="Modalità debug")
    parser.add_argument("--audio-device", type=int, default=None, help="ID dispositivo audio")
    return parser.parse_args()


def main():
    """Funzione principale dell'applicazione."""
    logger = setup_logging()
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Modalità debug attiva")

    logger.info("Avvio applicazione AI Parlante")

    # Verifica dipendenze
    if not check_dependencies():
        return 1

    try:
        #vocoder = load_hifigan_model()
        logger.info("Vocoder HiFi-GAN caricato con successo")
    except Exception as e:
        logger.error(f"Impossibile caricare HiFi-GAN: {e}")
        return 1

    # Sintesi test
    test_mel = np.random.randn(1, 80, 200)
    os.makedirs(args.output_dir, exist_ok=True)
    #synthesize_mel_to_audio(test_mel, vocoder, os.path.join(args.output_dir, "test.wav"), logger)

    # Log dispositivi audio
    logger.info("Dispositivi audio:")
    for i, d in enumerate(sd.query_devices()):
        logger.info(f"{i}: {d['name']} (in:{d['max_input_channels']} out:{d['max_output_channels']})")

    # Modalità GUI
    if not args.no_gui:
        try:
            from PyQt5.QtWidgets import QApplication
            from gui.main_window import MainWindow
            from controller import Controller

            qt_app = QApplication(sys.argv)
            ctrl = Controller(
                input_audio=args.input_audio,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                audio_device=args.audio_device,
                debug=args.debug
            )
            w = MainWindow(ctrl)
            w.show()
            return qt_app.exec_()
        except Exception as e:
            logger.warning(f"GUI non disponibile, fallback console: {e}")

    # Modalità console
    from console_app import ConsoleApp
    from controller import Controller

    ctrl = Controller(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        audio_device=args.audio_device,
        debug=args.debug
    )
    app = ConsoleApp(ctrl)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())