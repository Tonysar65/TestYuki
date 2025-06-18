"""
Modulo per la sintesi vocale.
Converte testo in parlato utilizzando modelli addestrati.
"""

import os
import logging
import numpy as np
import torch
import torchaudio
import librosa
from typing import Dict, Any, Optional, Callable


class Synthesizer:
    """Classe per la sintesi vocale."""

    def __init__(self, model_dir: str, debug: bool = False):
        self.logger = logging.getLogger("YukiAI.synthesizer")
        self.model_dir = model_dir
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 22050
        self.voice_model = None
        self.vocoder = None

        self._initialize_vocoder()
        self.logger.info(f"Synthesizer inizializzato (device={self.device})")

    def _initialize_vocoder(self):
        """Inizializza il vocoder per mel-to-audio."""
        try:
            import torchaudio.pipelines as pipelines
            self.vocoder = pipelines.HIFIGAN_VOCODER.get_model().to(self.device)
            self.sample_rate = pipelines.HIFIGAN_VOCODER.sample_rate
            self.logger.info(f"Vocoder HiFi-GAN caricato (sr={self.sample_rate})")
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Vocoder HiFi-GAN non disponibile: {e}")
            self.vocoder = None
            self.logger.info("Utilizzo di Griffin-Lim come fallback")

    def synthesize(self, text: str, model_name: str, model_path: str,
                  progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Sintetizza audio da testo."""
        try:
            if progress_callback:
                progress_callback(0.1)

            # Carica il modello vocale
            from voice_model_trainer import VoiceModel
            if self.voice_model is None:
                self.voice_model = VoiceModel(model_dir=self.model_dir, debug=self.debug)

            model_file = os.path.join(model_path, "model.pt")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Modello non trovato: {model_file}")

            if not self.voice_model.load_model(model_file):
                raise RuntimeError("Impossibile caricare il modello")

            if progress_callback:
                progress_callback(0.3)

            # Preprocessa testo e genera embedding
            processed_text = self._preprocess_text(text)
            embedding = np.random.randn(512)  # Embedding simulato

            if progress_callback:
                progress_callback(0.5)

            # Genera mel-spettrogramma
            mel_spectrogram = self.voice_model.synthesize(processed_text, embedding)

            if progress_callback:
                progress_callback(0.7)

            # Converti in audio
            waveform = self._mel_to_audio(mel_spectrogram)
            waveform = self._postprocess_audio(waveform)

            if progress_callback:
                progress_callback(1.0)

            self.logger.info("Sintesi completata")
            return waveform

        except Exception as e:
            self.logger.error(f"Errore nella sintesi: {e}")
            return np.zeros(self.sample_rate)  # 1 secondo di silenzio

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Preprocessa il testo per la sintesi."""
        text = text.lower()
        return ''.join(c for c in text if c.isalnum() or c.isspace() or c in ',.!?;:')

    def _mel_to_audio(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Converte mel-spettrogramma in audio."""
        try:
            mel_tensor = torch.FloatTensor(mel_spectrogram).to(self.device)
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0)

            if self.vocoder is not None:
                with torch.no_grad():
                    waveform = self.vocoder(mel_tensor)
                return waveform.cpu().numpy()[0]
            else:
                # Fallback: Griffin-Lim
                mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=1024, n_mels=mel_spectrogram.shape[0])
                mel_inverse = np.linalg.pinv(mel_basis)
                spec = np.dot(mel_inverse, mel_spectrogram)
                return librosa.griffinlim(spec, n_iter=32, hop_length=256)

        except Exception as e:
            self.logger.error(f"Errore nella conversione mel-to-audio: {e}")
            return np.zeros(self.sample_rate)

    @staticmethod
    def _postprocess_audio(waveform: np.ndarray) -> np.ndarray:
        """Post-processa l'audio generato."""
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.9
        return waveform

    def save_audio(self, waveform: np.ndarray, file_path: str) -> str:
        """Salva l'audio su file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)
            torchaudio.save(file_path, waveform_tensor, self.sample_rate)
            self.logger.info(f"Audio salvato: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio: {e}")
            raise

    def cleanup(self):
        """Libera le risorse."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.voice_model:
            self.voice_model.cleanup()
            self.voice_model = None

        self.vocoder = None