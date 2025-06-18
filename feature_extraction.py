"""
Modulo per l'estrazione delle caratteristiche audio.
Si occupa di estrarre le caratteristiche vocali dai file audio per l'addestramento del modello.
"""

import logging
import os
from typing import Dict, Any

import librosa
import numpy as np
import torch
import torchaudio


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
        self.logger = logging.getLogger("YukiAI.feature_extractor")
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
            y = audio_data["waveform"]
            sr = audio_data["sample_rate"]

            features = {
                "mel_spectrogram": self._extract_mel_spectrogram(y, sr),
                "mfcc": self._extract_mfcc(y, sr),
                "pitch": self._extract_pitch(y, sr),
                "energy": self._extract_energy(y),
                "prosody": self._extract_prosody(y, sr)
            }

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
        """
        y_tensor = torch.FloatTensor(y).to(self.device)

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)

        mel_spec = mel_spectrogram(y_tensor)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec_db.cpu().numpy()

    def _extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """
        Estrae i coefficienti cepstrali della frequenza mel (MFCC).
        """
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

    def _extract_pitch(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Estrae il pitch (F0) dalla forma d'onda.
        """
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )

        return {
            "f0": np.nan_to_num(pitch),
            "voiced_flag": voiced_flag,
            "voiced_probs": voiced_probs
        }

    def _extract_energy(self, y: np.ndarray) -> np.ndarray:
        """
        Estrae l'energia dalla forma d'onda.
        """
        energy = librosa.feature.rms(
            y=y,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return energy[0]

    def _extract_prosody(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Estrae le caratteristiche prosodiche dalla forma d'onda.
        """
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

        if len(onset_times) > 1:
            durations = np.diff(onset_times)
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
        """
        try:
            self.logger.info("Estrazione dell'embedding del parlante")
            y = audio_data["waveform"]
            sr = audio_data["sample_rate"]

            try:
                import torchaudio
                bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
                model = bundle.get_model().to(self.device)

                y_tensor = torch.FloatTensor(y).unsqueeze(0).to(self.device)

                if sr != bundle.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, bundle.sample_rate).to(self.device)
                    y_tensor = resampler(y_tensor)

                with torch.no_grad():
                    features, _ = model.extract_features(y_tensor)

                embedding = features[-1].mean(dim=1).cpu().numpy()
                self.logger.info(f"Embedding del parlante estratto: shape={embedding.shape}")
                return embedding

            except Exception as e:
                self.logger.warning(f"Errore durante l'estrazione dell'embedding con Wav2Vec2: {e}")
                self.logger.warning("Utilizzo di un metodo alternativo")

                mfcc = self._extract_mfcc(y, sr, n_mfcc=40)
                embedding = np.mean(mfcc, axis=1)
                self.logger.info(f"Embedding del parlante estratto (alternativo): shape={embedding.shape}")
                return embedding

        except Exception as e:
            self.logger.error(f"Errore durante l'estrazione dell'embedding del parlante: {e}")
            return np.array([])

    def save_features(self, audio_data: Dict[str, Any], file_path: str) -> str:
        """
        Salva le caratteristiche estratte in un file.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            features = audio_data.get("features", {})

            np.savez_compressed(file_path, **features)
            self.logger.info(f"Caratteristiche salvate: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio delle caratteristiche: {e}")
            raise

    def load_features(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Carica le caratteristiche da un file.
        """
        try:
            features = np.load(file_path, allow_pickle=True)
            features_dict = {key: features[key] for key in features.files}
            self.logger.info(f"Caratteristiche caricate: {file_path}")
            return features_dict
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento delle caratteristiche: {e}")
            raise