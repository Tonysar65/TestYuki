"""
Modulo per la clonazione vocale.
Si occupa di clonare una voce a partire da un file audio di riferimento.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from typing import Dict, Any, List, Tuple, Optional, Union, Callable


class VoiceCloner:
    """Classe per la clonazione vocale."""

    def __init__(self, model_dir: str, debug: bool = False):
        """
        Inizializza il clonatore vocale.

        Args:
            model_dir: Directory per i modelli
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.voice_cloner")
        self.model_dir = model_dir
        self.debug = debug

        # Verifica se CUDA è disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modello
        self.model = None

        # Vocoder
        self.vocoder = None

        # Parametri
        self.sample_rate = 22050

        self.logger.info(f"VoiceCloner inizializzato (device={self.device})")

        # Inizializza il modello
        self._initialize_model()

    def _initialize_model(self):
        """Inizializza il modello di clonazione vocale."""
        try:
            # Verifica se è disponibile un modello pre-addestrato
            model_path = os.path.join(self.model_dir, "voice_cloning", "model.pt")

            if os.path.exists(model_path):
                self.logger.info(f"Caricamento del modello pre-addestrato: {model_path}")
                self._load_model(model_path)
            else:
                self.logger.info("Nessun modello pre-addestrato trovato. Inizializzazione di un nuovo modello.")
                self._create_model()

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione del modello: {e}")
            raise

    def _create_model(self):
        """Crea un nuovo modello di clonazione vocale."""
        try:
            # In un'implementazione reale, si utilizzerebbe un modello più complesso
            # Per ora, utilizziamo un modello semplificato

            # Crea il modello
            self.model = VoiceCloningModel().to(self.device)

            # Crea il vocoder
            self._initialize_vocoder()

            self.logger.info("Nuovo modello creato")

        except Exception as e:
            self.logger.error(f"Errore durante la creazione del modello: {e}")
            raise

    def _load_model(self, model_path: str):
        """
        Carica un modello pre-addestrato.

        Args:
            model_path: Percorso del modello
        """
        try:
            # Carica il modello
            checkpoint = torch.load(model_path, map_location=self.device)

            # Crea il modello
            self.model = VoiceCloningModel().to(self.device)

            # Carica i pesi
            self.model.load_state_dict(checkpoint["model"])

            # Crea il vocoder
            self._initialize_vocoder()

            self.logger.info(f"Modello caricato: {model_path}")

        except Exception as e:
            self.logger.error(f"Errore durante il caricamento del modello: {e}")
            raise

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

    def clone_voice(self, audio_data: Dict[str, Any], model_name: str,
                    progress_callback: Optional[Callable[[float], None]] = None) -> str:
        """
        Clona una voce a partire da un file audio di riferimento.

        Args:
            audio_data: Dati audio
            model_name: Nome del modello
            progress_callback: Funzione di callback per il progresso

        Returns:
            str: Percorso del modello salvato
        """
        self.logger.info(f"Clonazione della voce: {model_name}")

        try:
            # Crea la directory del modello
            model_dir = os.path.join(self.model_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.1)

            # Estrai le caratteristiche
            features = audio_data.get("features", {})

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.2)

            # Addestra il modello
            self._train_model(features, progress_callback)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.8)

            # Salva il modello
            model_path = os.path.join(model_dir, "model.pt")
            self._save_model(model_path)

            # Salva i metadati
            metadata = {
                "model_name": model_name,
                "created_at": time.time(),
                "sample_rate": self.sample_rate,
                "device": str(self.device)
            }

            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(1.0)

            self.logger.info(f"Modello salvato: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Errore durante la clonazione della voce: {e}")
            raise

    def _train_model(self, features: Dict[str, np.ndarray],
                     progress_callback: Optional[Callable[[float], None]] = None):
        """
        Addestra il modello di clonazione vocale.

        Args:
            features: Caratteristiche audio
            progress_callback: Funzione di callback per il progresso
        """
        try:
            # Estrai le caratteristiche
            mel_spectrogram = features.get("mel_spectrogram", None)
            mfcc = features.get("mfcc", None)
            pitch = features.get("pitch", {}).get("f0", None)
            energy = features.get("energy", None)

            if mel_spectrogram is None or mfcc is None:
                self.logger.error("Caratteristiche mancanti")
                raise ValueError("Caratteristiche mancanti")

            # Converti in tensori PyTorch
            mel_tensor = torch.FloatTensor(mel_spectrogram).to(self.device)
            mfcc_tensor = torch.FloatTensor(mfcc).to(self.device)

            # Aggiungi la dimensione del batch
            mel_tensor = mel_tensor.unsqueeze(0)  # [1, n_mels, time]
            mfcc_tensor = mfcc_tensor.unsqueeze(0)  # [1, n_mfcc, time]

            # Trasponi per ottenere [batch, time, features]
            mel_tensor = mel_tensor.transpose(1, 2)  # [1, time, n_mels]
            mfcc_tensor = mfcc_tensor.transpose(1, 2)  # [1, time, n_mfcc]

            # Imposta il modello in modalità di addestramento
            self.model.train()

            # Ottimizzatore
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            # Funzione di perdita
            criterion = nn.MSELoss()

            # Numero di epoche
            num_epochs = 100

            # Addestramento
            for epoch in range(num_epochs):
                # Azzera i gradienti
                optimizer.zero_grad()

                # Forward pass
                output = self.model(mfcc_tensor)

                # Calcola la perdita
                loss = criterion(output, mel_tensor)

                # Backward pass
                loss.backward()

                # Aggiorna i pesi
                optimizer.step()

                # Log
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    self.logger.info(f"Epoca {epoch + 1}/{num_epochs}, Perdita: {loss.item():.4f}")

                # Aggiorna il progresso
                if progress_callback:
                    progress = 0.2 + (epoch + 1) / num_epochs * 0.6
                    progress_callback(progress)

            # Imposta il modello in modalità di valutazione
            self.model.eval()

            self.logger.info("Addestramento completato")

        except Exception as e:
            self.logger.error(f"Errore durante l'addestramento del modello: {e}")
            raise

    def _save_model(self, model_path: str):
        """
        Salva il modello.

        Args:
            model_path: Percorso del modello
        """
        try:
            # Crea la directory
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Salva il modello
            torch.save({
                "model": self.model.state_dict(),
                "sample_rate": self.sample_rate
            }, model_path)

            self.logger.info(f"Modello salvato: {model_path}")

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio del modello: {e}")
            raise

    def synthesize(self, text_encoding: torch.Tensor, model_path: str,
                   progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Sintetizza il parlato a partire dal testo utilizzando il modello di clonazione vocale.

        Args:
            text_encoding: Encoding del testo
            model_path: Percorso del modello
            progress_callback: Funzione di callback per il progresso

        Returns:
            np.ndarray: Forma d'onda audio
        """
        self.logger.info("Sintesi vocale con modello clonato")

        try:
            # Carica il modello se necessario
            if model_path != getattr(self, "current_model_path", None):
                self._load_model(model_path)
                self.current_model_path = model_path

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.1)

            # Imposta il modello in modalità di valutazione
            self.model.eval()

            # Forward pass
            with torch.no_grad():
                mel_spectrogram = self.model(text_encoding)

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.5)

            # Converti il mel-spettrogramma in forma d'onda
            waveform = self._mel_to_audio(mel_spectrogram[0])

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(0.9)

            # Normalizza
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.9

            # Aggiorna il progresso
            if progress_callback:
                progress_callback(1.0)

            self.logger.info("Sintesi completata")
            return waveform

        except Exception as e:
            self.logger.error(f"Errore durante la sintesi: {e}")
            raise

    def _mel_to_audio(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        Converte un mel-spettrogramma in forma d'onda audio.

        Args:
            mel_spectrogram: Mel-spettrogramma

        Returns:
            np.ndarray: Forma d'onda audio
        """
        try:
            # Aggiungi la dimensione del batch se necessario
            if mel_spectrogram.dim() == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)

            # Utilizza il vocoder se disponibile
            if self.vocoder is not None:
                with torch.no_grad():
                    waveform = self.vocoder(mel_spectrogram)

                # Converti in numpy
                waveform = waveform.cpu().numpy()[0]
            else:
                # Fallback: Griffin-Lim
                self.logger.warning("Utilizzo dell'algoritmo Griffin-Lim come fallback")

                # Converti in numpy
                mel_np = mel_spectrogram.cpu().numpy()[0].T

                # Converti il mel-spettrogramma in spettrogramma
                import librosa
                mel_basis = librosa.filters.mel(sr=self.sample_rate, n_fft=1024, n_mels=mel_np.shape[0])
                mel_inverse = np.linalg.pinv(mel_basis)
                spec = np.dot(mel_inverse, mel_np)

                # Applica Griffin-Lim
                waveform = librosa.griffinlim(spec, n_iter=32, hop_length=256)

            return waveform

        except Exception as e:
            self.logger.error(f"Errore durante la conversione mel-to-audio: {e}")

            # Restituisci una forma d'onda vuota
            return np.zeros(self.sample_rate)  # 1 secondo di silenzio

    def cleanup(self):
        """Libera le risorse."""
        # Libera la memoria CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Libera il modello
        self.model = None

        # Libera il vocoder
        self.vocoder = None


class VoiceCloningModel(nn.Module):
    """Modello per la clonazione vocale."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 256, output_dim: int = 80):
        """
        Inizializza il modello.

        Args:
            input_dim: Dimensione dell'input (numero di coefficienti MFCC)
            hidden_dim: Dimensione dello stato nascosto
            output_dim: Dimensione dell'output (numero di bande mel)
        """
        super(VoiceCloningModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim] (MFCC features)

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, output_dim] (mel-spectrogram)
        """
        # Get batch dimensions
        batch_size, seq_len, _ = x.size()

        # Process each time step
        encoded = []
        for t in range(seq_len):
            # Get current time step
            x_t = x[:, t, :]

            # Encoder
            h = self.encoder(x_t)

            # Store encoded representation
            encoded.append(h)

        # Stack all time steps
        encoded = torch.stack(encoded, dim=1)  # [batch_size, seq_len, hidden_dim]

        # Decode each time step
        output = []
        for t in range(seq_len):
            # Get current encoded representation
            h_t = encoded[:, t, :]

            # Decoder
            out_t = self.decoder(h_t)

            # Store output
            output.append(out_t)

        # Stack all outputs
        output = torch.stack(output, dim=1)  # [batch_size, seq_len, output_dim]

        return output

    def init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for layer in [self.encoder, self.decoder]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)