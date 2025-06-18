"""
Modulo per l'addestramento del modello vocale.
Si occupa di addestrare un modello di sintesi vocale basato sulle caratteristiche estratte.
"""

import os
import logging
import time
import json
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union, Callable


class VoiceEncoder(nn.Module):
    """Encoder per la voce del parlante."""

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256,
                 embedding_dim: int = 512, num_layers: int = 3):
        super(VoiceEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(hidden_dim * 2, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        embedding = self.linear(hidden)
        embedding = self.relu(embedding)
        return F.normalize(embedding, p=2, dim=1)


class TextEncoder(nn.Module):
    """Encoder per il testo."""

    def __init__(self, vocab_size: int, embedding_dim: int = 512,
                 hidden_dim: int = 512, num_layers: int = 2):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.linear(output)
        return self.relu(output)


class Decoder(nn.Module):
    """Decoder per la sintesi vocale."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024,
                 output_dim: int = 80, num_layers: int = 3):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.linear(output)


class VoiceModel:
    """Classe per l'addestramento e l'utilizzo del modello vocale."""

    def __init__(self, model_dir: str, debug: bool = False):
        self.logger = logging.getLogger("YukiAI.voice_model")
        self.model_dir = model_dir
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parametri del modello
        self.input_dim = 80
        self.hidden_dim = 512
        self.embedding_dim = 512
        self.vocab_size = 1000

        # Modelli
        self.voice_encoder = None
        self.text_encoder = None
        self.decoder = None

        # Ottimizzatori
        self.voice_encoder_optimizer = None
        self.text_encoder_optimizer = None
        self.decoder_optimizer = None

        # Vocabolario
        self.vocab = None

        self.logger.info(f"VoiceModel inizializzato (device={self.device})")

    def train(self, features: Dict[str, Any], model_name: str, model_path: str,
              progress_callback: Optional[Callable[[float], None]] = None) -> str:
        """Addestra un modello vocale."""
        try:
            os.makedirs(model_path, exist_ok=True)

            # Estrai e prepara i dati
            mel_spectrogram = torch.FloatTensor(features["features"]["mel_spectrogram"]).to(self.device)
            mel_spectrogram = mel_spectrogram.unsqueeze(0).transpose(1, 2)

            # Inizializza i modelli
            self._initialize_models()

            # Addestra il modello
            self._train_model(
                mel_spectrogram_tensor=mel_spectrogram,
                progress_callback=progress_callback
            )

            # Salva il modello
            model_file = os.path.join(model_path, "model.pt")
            self._save_model(model_file)

            # Salva i metadati
            metadata = {
                "model_name": model_name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "vocab_size": self.vocab_size,
                "created_at": time.time(),
                "device": str(self.device)
            }

            metadata_file = os.path.join(model_path, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Modello salvato: {model_file}")
            return model_file

        except Exception as e:
            self.logger.error(f"Errore durante l'addestramento: {e}")
            raise

    def _initialize_models(self):
        """Inizializza i modelli e gli ottimizzatori."""
        self.voice_encoder = VoiceEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        self.text_encoder = TextEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        self.decoder = Decoder(
            input_dim=self.hidden_dim + self.embedding_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.input_dim
        ).to(self.device)

        self.voice_encoder_optimizer = optim.Adam(self.voice_encoder.parameters(), lr=0.001)
        self.text_encoder_optimizer = optim.Adam(self.text_encoder.parameters(), lr=0.001)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.001)

    def _train_model(self, mel_spectrogram_tensor: torch.Tensor,
                    num_epochs: int = 100,
                    progress_callback: Optional[Callable[[float], None]] = None):
        """Addestra il modello."""
        self.voice_encoder.train()
        self.text_encoder.train()
        self.decoder.train()

        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            self.voice_encoder_optimizer.zero_grad()
            self.text_encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Forward pass
            voice_embedding = self.voice_encoder(mel_spectrogram_tensor)

            batch_size = mel_spectrogram_tensor.size(0)
            seq_len = mel_spectrogram_tensor.size(1)
            text_input = torch.randint(0, self.vocab_size, (batch_size, seq_len // 2)).to(self.device)
            text_encoding = self.text_encoder(text_input)

            voice_embedding_expanded = voice_embedding.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
            decoder_input = torch.cat([text_encoding, voice_embedding_expanded], dim=2)

            output = self.decoder(decoder_input)
            target = mel_spectrogram_tensor[:, :output.size(1), :]

            # Calcola e backpropaga la perdita
            loss = criterion(output, target)
            loss.backward()

            self.voice_encoder_optimizer.step()
            self.text_encoder_optimizer.step()
            self.decoder_optimizer.step()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoca {epoch + 1}/{num_epochs}, Perdita: {loss.item():.4f}")

            if progress_callback:
                progress_callback((epoch + 1) / num_epochs)

    def _save_model(self, model_file: str):
        """Salva il modello su disco."""
        model_dict = {
            "voice_encoder": self.voice_encoder.state_dict(),
            "text_encoder": self.text_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size
        }

        torch.save(model_dict, model_file)

    def load_model(self, model_file: str) -> bool:
        """Carica un modello da disco."""
        try:
            model_dict = torch.load(model_file, map_location=self.device)

            self.input_dim = model_dict["input_dim"]
            self.hidden_dim = model_dict["hidden_dim"]
            self.embedding_dim = model_dict["embedding_dim"]
            self.vocab_size = model_dict["vocab_size"]

            self._initialize_models()

            self.voice_encoder.load_state_dict(model_dict["voice_encoder"])
            self.text_encoder.load_state_dict(model_dict["text_encoder"])
            self.decoder.load_state_dict(model_dict["decoder"])

            self.voice_encoder.eval()
            self.text_encoder.eval()
            self.decoder.eval()

            self.logger.info(f"Modello caricato: {model_file}")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante il caricamento: {e}")
            return False

    def get_embedding(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Calcola l'embedding vocale."""
        try:
            mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0).to(self.device)
            mel_tensor = mel_tensor.transpose(1, 2)

            with torch.no_grad():
                embedding = self.voice_encoder(mel_tensor)

            return embedding.cpu().numpy()[0]

        except Exception as e:
            self.logger.error(f"Errore nel calcolo dell'embedding: {e}")
            raise

    def synthesize(self, text: Union[str, List[int]], voice_embedding: np.ndarray,
                  progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Sintetizza audio da testo e embedding vocale."""
        try:
            if progress_callback:
                progress_callback(0.1)

            # Prepara l'input
            if isinstance(text, str):
                text_tokens = [ord(c) % self.vocab_size for c in text]
            else:
                text_tokens = text

            text_tensor = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)
            voice_tensor = torch.FloatTensor(voice_embedding).unsqueeze(0).to(self.device)

            if progress_callback:
                progress_callback(0.4)

            # Sintesi
            with torch.no_grad():
                text_encoding = self.text_encoder(text_tensor)
                voice_embedding_expanded = voice_tensor.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
                decoder_input = torch.cat([text_encoding, voice_embedding_expanded], dim=2)
                mel_output = self.decoder(decoder_input)

            mel_output = mel_output.squeeze(0).cpu().numpy().T

            if progress_callback:
                progress_callback(1.0)

            return mel_output

        except Exception as e:
            self.logger.error(f"Errore durante la sintesi: {e}")
            raise

    @staticmethod
    def cleanup():
        """Libera le risorse."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()