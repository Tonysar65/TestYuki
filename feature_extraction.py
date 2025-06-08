#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modulo per l'estrazione delle caratteristiche audio.
Si occupa di estrarre le caratteristiche vocali dai file audio per l'addestramento del modello.
"""

import os
import logging
import numpy as np
import librosa
import torch
import torchaudio
from typing import Dict, Any, List, Tuple, Optional, Union


class FeatureExtractor:
    """Classe per l'estrazione delle caratteristiche audio."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256,
                 n_mels: int = 80, debug: bool = False):
        """
        Inizializza l'estrattore di caratteristiche.

        Args:
            n_fft: Dimensione della FFT
            hop_length: Lunghezza del salto per la STFT
            n_mels: Numero di bande mel
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.feature_extractor")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.debug = debug

        # Verifica se CUDA è disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"FeatureExtractor inizializzato (device={self.device}, "
                         f"n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})")

    def extract_features(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estrae le caratteristiche audio da un file audio.

        Args:
            audio_data: Dizionario contenente i dati audio e i metadati

        Returns:
            Dict[str, Any]: Dizionario contenente le caratteristiche estratte
        """
        self.logger.info("Estrazione delle caratteristiche audio")

        try:
            # Estrai la forma d'onda e la frequenza di campionamento
            y = audio_data["waveform"]
            sr = audio_data["sample_rate"]

            # Estrai le caratteristiche
            features = {}

            # Mel-spettrogramma
            features["mel_spectrogram"] = self._extract_mel_spectrogram(y, sr)

            # MFCC
            features["mfcc"] = self._extract_mfcc(y, sr)

            # Pitch (F0)
            features["pitch"] = self._extract_pitch(y, sr)

            # Energia
            features["energy"] = self._extract_energy(y)

            # Caratteristiche prosodiche
            features["prosody"] = self._extract_prosody(y, sr)

            # Aggiungi le caratteristiche al dizionario audio
            audio_data["features"] = features

            self.logger.info("Estrazione delle caratteristiche completata")

            if self.debug:
                for name, feature in features.items():
                    if isinstance(feature, np.ndarray):
                        self.logger.debug(f"Caratteristica {name}: shape={feature.shape}")
                    elif isinstance(feature, dict):
                        for subname, subfeature in feature.items():
                            if isinstance(subfeature, np.ndarray):
                                self.logger.debug(f"Caratteristica {name}.{subname}: shape={subfeature.shape}")

            return audio_data

        except Exception as e:
            self.logger.error(f"Errore durante l'estrazione delle caratteristiche: {e}")
            raise

    def _extract_mel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Estrae il mel-spettrogramma dalla forma d'onda.

        Args:
            y: Forma d'onda audio
            sr: Frequenza di campionamento

        Returns:
            np.ndarray: Mel-spettrogramma
        """
        # Converti in tensore PyTorch
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Crea il trasformatore mel-spettrogramma
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)

        # Estrai il mel-spettrogramma
        mel_spec = mel_spectrogram(y_tensor)

        # Converti in decibel
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Converti in numpy
        mel_spec_db_np = mel_spec_db.cpu().numpy()

        return mel_spec_db_np

    def _extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """
        Estrae i coefficienti cepstrali della frequenza mel (MFCC) dalla forma d'onda.

        Args:
            y: Forma d'onda audio
            sr: Frequenza di campionamento
            n_mfcc: Numero di coefficienti MFCC

        Returns:
            np.ndarray: MFCC
        """
        # Estrai gli MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Normalizza gli MFCC
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

        return mfcc

    def _extract_pitch(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Estrae il pitch (F0) dalla forma d'onda.

        Args:
            y: Forma d'onda audio
            sr: Frequenza di campionamento

        Returns:
            Dict[str, np.ndarray]: Dizionario contenente il pitch e la confidenza
        """
        # Estrai il pitch
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )

        # Sostituisci i valori NaN con 0
        pitch = np.nan_to_num(pitch)

        return {
            "f0": pitch,
            "voiced_flag": voiced_flag,
            "voiced_probs": voiced_probs
        }

    def _extract_energy(self, y: np.ndarray) -> np.ndarray:
        """
        Estrae l'energia dalla forma d'onda.

        Args:
            y: Forma d'onda audio

        Returns:
            np.ndarray: Energia
        """
        # Calcola l'energia RMS
        energy = librosa.feature.rms(
            y=y,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )

        return energy[0]

    def _extract_prosody(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Estrae le caratteristiche prosodiche dalla forma d'onda.

        Args:
            y: Forma d'onda audio
            sr: Frequenza di campionamento

        Returns:
            Dict[str, np.ndarray]: Dizionario contenente le caratteristiche prosodiche
        """
        # Calcola il tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)

        # Calcola il ritmo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Calcola la durata delle sillabe (approssimazione)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

        # Calcola le durate tra gli onset
        if len(onset_times) > 1:
            durations = np.diff(onset_times)
            # Aggiungi l'ultima durata (approssimazione)
            durations = np.append(durations, durations[-1])
        else:
            durations = np.array([0.0])

        return {
            "tempo": tempo,
            "onset_strength": onset_env,
            "onset_times": onset_times,
            "durations": durations
        }

    def extract_speaker_embedding(self, audio_data: Dict[str, Any]) -> np.ndarray:
        """
        Estrae l'embedding del parlante dalla forma d'onda.
        Questo è utile per la clonazione vocale.

        Args:
            audio_data: Dizionario contenente i dati audio e i metadati

        Returns:
            np.ndarray: Embedding del parlante
        """
        try:
            # Prova a importare il modello di embedding del parlante
            import torch
            import torchaudio

            self.logger.info("Estrazione dell'embedding del parlante")

            # Estrai la forma d'onda e la frequenza di campionamento
            y = audio_data["waveform"]
            sr = audio_data["sample_rate"]

            # Converti in tensore PyTorch
            y_tensor = torch.FloatTensor(y).unsqueeze(0).to(self.device)

            # Carica il modello pre-addestrato (se disponibile)
            try:
                # Prova a caricare il modello Wav2Vec2
                bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
                model = bundle.get_model().to(self.device)

                # Ricampiona se necessario
                if sr != bundle.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, bundle.sample_rate).to(self.device)
                    y_tensor = resampler(y_tensor)

                # Estrai le caratteristiche
                with torch.no_grad():
                    features, _ = model.extract_features(y_tensor)

                # Usa l'ultimo layer come embedding
                embedding = features[-1].mean(dim=1).cpu().numpy()

                self.logger.info(f"Embedding del parlante estratto: shape={embedding.shape}")
                return embedding

            except Exception as e:
                self.logger.warning(f"Errore durante l'estrazione dell'embedding con Wav2Vec2: {e}")
                self.logger.warning("Utilizzo di un metodo alternativo")

                # Metodo alternativo: usa gli MFCC come embedding
                mfcc = self._extract_mfcc(y, sr, n_mfcc=40)
                embedding = np.mean(mfcc, axis=1)

                self.logger.info(f"Embedding del parlante estratto (alternativo): shape={embedding.shape}")
                return embedding

        except Exception as e:
            self.logger.error(f"Errore durante l'estrazione dell'embedding del parlante: {e}")
            # Restituisci un embedding vuoto
            return np.array([])

    def save_features(self, audio_data: Dict[str, Any], file_path: str) -> str:
        """
        Salva le caratteristiche estratte in un file.

        Args:
            audio_data: Dizionario contenente i dati audio e le caratteristiche
            file_path: Percorso del file di output

        Returns:
            str: Percorso del file salvato
        """
        try:
            # Assicura che la directory esista
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Estrai le caratteristiche
            features = audio_data.get("features", {})

            # Salva le caratteristiche
            np.savez_compressed(
                file_path,
                **features
            )

            self.logger.info(f"Caratteristiche salvate: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio delle caratteristiche: {e}")
            raise

    def load_features(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Carica le caratteristiche da un file.

        Args:
            file_path: Percorso del file

        Returns:
            Dict[str, np.ndarray]: Dizionario contenente le caratteristiche
        """
        try:
            # Carica le caratteristiche
            features = np.load(file_path, allow_pickle=True)

            # Converti in dizionario
            features_dict = {key: features[key] for key in features.files}

            self.logger.info(f"Caratteristiche caricate: {file_path}")
            return features_dict

        except Exception as e:
            self.logger.error(f"Errore durante il caricamento delle caratteristiche: {e}")
            raise

