#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modulo per il preprocessing audio.
Si occupa di caricare, normalizzare e preparare i file audio per l'elaborazione.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, Tuple, Optional, Union


class AudioProcessor:
    """Classe per il preprocessing audio."""

    def __init__(self, sample_rate: int = 22050, debug: bool = False):
        """
        Inizializza il processore audio.

        Args:
            sample_rate: Frequenza di campionamento target
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.audio_processor")
        self.sample_rate = sample_rate
        self.debug = debug
        self.logger.info(f"AudioProcessor inizializzato (sample_rate={sample_rate})")

    def load_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Carica un file audio e lo prepara per l'elaborazione.

        Args:
            file_path: Percorso del file audio

        Returns:
            Dict[str, Any]: Dizionario contenente i dati audio e i metadati
        """
        self.logger.info(f"Caricamento del file audio: {file_path}")

        try:
            # Carica il file audio
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Normalizza l'audio
            y = self._normalize_audio(y)

            # Rimuovi il silenzio iniziale e finale
            y = self._trim_silence(y)

            # Calcola la durata in secondi
            duration = librosa.get_duration(y=y, sr=sr)

            # Crea il dizionario di output
            audio_data = {
                "waveform": y,
                "sample_rate": sr,
                "duration": duration,
                "file_path": file_path,
                "file_name": os.path.basename(file_path)
            }

            self.logger.info(f"File audio caricato: {os.path.basename(file_path)} "
                             f"(durata: {duration:.2f}s, sr: {sr}Hz)")

            if self.debug:
                self.logger.debug(f"Forma d'onda: shape={y.shape}, min={y.min():.4f}, max={y.max():.4f}")

            return audio_data

        except Exception as e:
            self.logger.error(f"Errore durante il caricamento del file audio: {e}")
            raise

    def save_audio(self, audio_data: Union[Dict[str, Any], np.ndarray],
                   file_path: str, sample_rate: Optional[int] = None) -> str:
        """
        Salva i dati audio in un file.

        Args:
            audio_data: Dati audio (dizionario o array numpy)
            file_path: Percorso del file di output
            sample_rate: Frequenza di campionamento (se non specificata, usa quella dei dati o quella predefinita)

        Returns:
            str: Percorso del file salvato
        """
        try:
            # Estrai la forma d'onda e la frequenza di campionamento
            if isinstance(audio_data, dict):
                waveform = audio_data["waveform"]
                sr = audio_data.get("sample_rate", self.sample_rate)
            else:
                waveform = audio_data
                sr = sample_rate or self.sample_rate

            # Assicura che la directory esista
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Salva il file audio
            sf.write(file_path, waveform, sr)

            self.logger.info(f"File audio salvato: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio del file audio: {e}")
            raise

    @staticmethod
    def _normalize_audio(y: np.ndarray) -> np.ndarray:
        """
        Normalizza l'audio per avere un'ampiezza massima di 1.0.

        Args:
            y: Forma d'onda audio

        Returns:
            np.ndarray: Forma d'onda normalizzata
        """
        # Evita la divisione per zero
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def _trim_silence(self, y: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """
        Rimuove il silenzio iniziale e finale dalla forma d'onda.

        Args:
            y: Forma d'onda audio
            threshold_db: Soglia in dB per il rilevamento del silenzio

        Returns:
            np.ndarray: Forma d'onda senza silenzio
        """
        try:
            # Rimuovi il silenzio
            y_trimmed, _ = librosa.effects.trim(y, top_db=-threshold_db)
            return y_trimmed
        except Exception as e:
            self.logger.warning(f"Errore durante la rimozione del silenzio: {e}. Utilizzo dell'audio originale.")
            return y

    @staticmethod
    def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Ricampiona l'audio a una nuova frequenza di campionamento.

        Args:
            y: Forma d'onda audio
            orig_sr: Frequenza di campionamento originale
            target_sr: Frequenza di campionamento target

        Returns:
            np.ndarray: Forma d'onda ricampionata
        """
        if orig_sr == target_sr:
            return y

        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

    def split_audio(self, audio_data: Dict[str, Any],
                    segment_length: float = 10.0,
                    overlap: float = 1.0) -> list:
        """
        Divide l'audio in segmenti di lunghezza specificata.

        Args:
            audio_data: Dati audio
            segment_length: Lunghezza dei segmenti in secondi
            overlap: Sovrapposizione tra segmenti in secondi

        Returns:
            list: Lista di segmenti audio
        """
        y = audio_data["waveform"]
        sr = audio_data["sample_rate"]

        # Converti da secondi a campioni
        segment_samples = int(segment_length * sr)
        overlap_samples = int(overlap * sr)
        hop_samples = segment_samples - overlap_samples

        # Calcola il numero di segmenti
        num_segments = max(1, int(np.ceil((len(y) - overlap_samples) / hop_samples)))

        segments = []
        for i in range(num_segments):
            start = i * hop_samples
            end = min(start + segment_samples, len(y))

            # Se l'ultimo segmento è troppo corto, estendilo all'indietro
            if end - start < segment_samples and i > 0:
                start = max(0, end - segment_samples)

            segment = y[start:end]

            # Crea un nuovo dizionario per il segmento
            segment_data = audio_data.copy()
            segment_data["waveform"] = segment
            segment_data["duration"] = len(segment) / sr
            segment_data["segment_index"] = i

            segments.append(segment_data)

        self.logger.info(f"Audio diviso in {len(segments)} segmenti "
                         f"(lunghezza: {segment_length}s, sovrapposizione: {overlap}s)")

        return segments

    def apply_noise_reduction(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Applica la riduzione del rumore alla forma d'onda.

        Args:
            y: Forma d'onda audio
            sr: Frequenza di campionamento

        Returns:
            np.ndarray: Forma d'onda con rumore ridotto
        """
        try:
            import noisereduce as nr

            # Applica la riduzione del rumore
            y_reduced = nr.reduce_noise(y=y, sr=sr)

            self.logger.info("Riduzione del rumore applicata")
            return y_reduced

        except ImportError:
            self.logger.warning("Libreria noisereduce non disponibile. Utilizzo dell'audio originale.")
            return y
        except Exception as e:
            self.logger.warning(f"Errore durante la riduzione del rumore: {e}. Utilizzo dell'audio originale.")
            return y

