"""
Modulo per la sintesi vocale.
Si occupa di convertire il testo in parlato utilizzando il modello vocale addestrato.
"""

import os
import logging
import time
import json

import librosa
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, List, Tuple, Optional, Union, Callable


class Synthesizer:
    """Classe per la sintesi vocale."""

    def __init__(self, model_dir: str, debug: bool = False):
        """
        Inizializza il sintetizzatore.

        Args:
            model_dir: Directory per i modelli
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.synthesizer")
        self.model_dir = model_dir
        self.debug = debug

        # Verifica se CUDA è disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modello vocale
        self.voice_model = None

        # Vocoder (converte mel-spettrogramma in forma d'onda)
        self.vocoder = None

        # Parametri
        self.sample_rate = 22050

        self.logger.info(f"Synthesizer inizializzato (device={self.device})")

        # Inizializza il vocoder
        self._initialize_vocoder()

    def _initialize_vocoder(self):
        """Inizializza il vocoder."""
        try:
            # Prova a caricare il vocoder HiFi-GAN
            self.logger.info("Inizializzazione del vocoder HiFi-GAN")

            # Verifica se PyTorch include il vocoder pre-addestrato
            try:
                import torchaudio.pipelines as pipelines

                # Carica il vocoder
                self.vocoder = pipelines.HIFIGAN_VOCODER.get_model().to(self.device)
                self.sample_rate = pipelines.HIFIGAN_VOCODER.sample_rate

                self.logger.info(f"Vocoder HiFi-GAN caricato (sample_rate={self.sample_rate})")
            except (ImportError, AttributeError):
                self.logger.warning("Vocoder HiFi-GAN non disponibile in torchaudio")
                self.logger.warning("Utilizzo di un vocoder alternativo")

                # Implementazione alternativa: Griffin-Lim
                self.vocoder = None
                self.logger.info("Vocoder Griffin-Lim sarà utilizzato come fallback")

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione del vocoder: {e}")
            self.vocoder = None

    def synthesize(self, text: str, model_name: str, model_path: str,
                   progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Sintetizza il parlato a partire dal testo utilizzando il modello vocale specificato.

        Args:
            text: Testo da sintetizzare
            model_name: Nome del modello
            model_path: Percorso del modello
            progress_callback: Funzione di callback per il progresso

        Returns:
            np.ndarray: Forma d'onda audio
        """
        self.logger.info(f"Sintesi vocale: '{text}' con modello {model_name}")

        try:
            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.1)

            # Carica il modello vocale
            from voice_model_trainer import VoiceModel

            if self.voice_model is None:
                self.voice_model = VoiceModel(model_dir=self.model_dir, debug=self.debug)

            # Carica il modello
            model_file = os.path.join(model_path, "model.pt")
            if not os.path.exists(model_file):
                self.logger.error(f"Modello non trovato: {model_file}")
                return np.array([])

            self.voice_model.load_model(model_file)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.2)

            # Preprocessa il testo
            processed_text = self._preprocess_text(text)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.3)

            # Genera l'embedding vocale (simulato per questo esempio)
            # In un'implementazione reale, si utilizzerebbe un embedding estratto dal file audio di riferimento
            embedding = np.random.randn(512)  # Dimensione dell'embedding

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.4)

            # Sintetizza il mel-spettrogramma
            mel_spectrogram = self.voice_model.synthesize(processed_text, embedding)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.6)

            # Converti il mel-spettrogramma in forma d'onda
            waveform = self._mel_to_audio(mel_spectrogram)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.8)

            # Applica il post-processing
            waveform = self._postprocess_audio(waveform)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(1.0)

            self.logger.info("Sintesi vocale completata")
            return waveform

        except Exception as e:
            self.logger.error(f"Errore durante la sintesi vocale: {e}")
            return np.array([])

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocessa il testo per la sintesi.

        Args:
            text: Testo da preprocessare

        Returns:
            str: Testo preprocessato
        """
        # Normalizza il testo
        text = text.lower()

        # Rimuovi caratteri speciali
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ',.!?;:')

        return text

    def _mel_to_audio(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Converte un mel-spettrogramma in forma d'onda audio.

        Args:
            mel_spectrogram: Mel-spettrogramma

        Returns:
            np.ndarray: Forma d'onda audio
        """
        try:
            # Converti in tensore PyTorch
            mel_tensor = torch.FloatTensor(mel_spectrogram).to(self.device)

            # Aggiungi la dimensione del batch se necessario
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0)

            # Utilizza il vocoder se disponibile
            if self.vocoder is not None:
                with torch.no_grad():
                    waveform = self.vocoder(mel_tensor)

                # Converti in numpy
                waveform = waveform.cpu().numpy()[0]
            else:
                # Fallback: Griffin-Lim
                self.logger.warning("Utilizzo dell'algoritmo Griffin-Lim come fallback")

                # Converti il mel-spettrogramma in spettrogramma
                mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=1024, n_mels=mel_spectrogram.shape[0])
                mel_inverse = np.linalg.pinv(mel_basis)
                spec = np.dot(mel_inverse, mel_spectrogram)

                # Applica Griffin-Lim
                waveform = librosa.griffinlim(spec, n_iter=32, hop_length=256)

            return waveform

        except Exception as e:
            self.logger.error(f"Errore durante la conversione mel-to-audio: {e}")

            # Restituisci una forma d'onda vuota
            return np.zeros(self.sample_rate)  # 1 secondo di silenzio

    @staticmethod
    def _postprocess_audio(waveform: np.ndarray) -> np.ndarray:
        """
        Applica il post-processing alla forma d'onda audio.

        Args:
            waveform: Forma d'onda audio

        Returns:
            np.ndarray: Forma d'onda audio post-processata
        """
        # Normalizza
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.9  # Lascia un po' di headroom

        return waveform

    def save_audio(self, waveform: np.ndarray, file_path: str) -> str:
        """
        Salva la forma d'onda audio in un file.

        Args:
            waveform: Forma d'onda audio
            file_path: Percorso del file

        Returns:
            str: Percorso del file salvato
        """
        try:
            # Assicura che la directory esista
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Converti in tensore PyTorch
            waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)

            # Salva il file audio
            torchaudio.save(file_path, waveform_tensor, self.sample_rate)

            self.logger.info(f"File audio salvato: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio del file audio: {e}")
            raise

    def cleanup(self):
        """Libera le risorse."""
        # Libera la memoria CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Libera le risorse del modello vocale
        if self.voice_model:
            self.voice_model.cleanup()
            self.voice_model = None

        # Libera le risorse del vocoder
        self.vocoder = None

