"""
Modulo per l'addestramento del modello vocale.
Si occupa di addestrare un modello di sintesi vocale basato sulle caratteristiche estratte.
"""

import os
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional, Union, Callable


class VoiceEncoder(nn.Module):
    """
    Encoder per la voce del parlante.
    Converte le caratteristiche audio in un embedding del parlante.
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256,
                 embedding_dim: int = 512, num_layers: int = 3):
        """
        Inizializza l'encoder.

        Args:
            input_dim: Dimensione dell'input (numero di bande mel)
            hidden_dim: Dimensione dello stato nascosto
            embedding_dim: Dimensione dell'embedding
            num_layers: Numero di layer LSTM
        """
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
        """
        Forward pass.

        Args:
            x: Input di forma [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Embedding di forma [batch_size, embedding_dim]
        """
        # LSTM
        output, (hidden, _) = self.lstm(x)

        # Prendi l'ultimo stato nascosto di entrambe le direzioni
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # Proietta nell'embedding
        embedding = self.linear(hidden)
        embedding = self.relu(embedding)

        # Normalizza l'embedding
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class TextEncoder(nn.Module):
    """
    Encoder per il testo.
    Converte il testo in una rappresentazione latente.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 512,
                 hidden_dim: int = 512, num_layers: int = 2):
        """
        Inizializza l'encoder.

        Args:
            vocab_size: Dimensione del vocabolario
            embedding_dim: Dimensione dell'embedding
            hidden_dim: Dimensione dello stato nascosto
            num_layers: Numero di layer LSTM
        """
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
        """
        Forward pass.

        Args:
            x: Input di forma [batch_size, seq_len]

        Returns:
            torch.Tensor: Rappresentazione di forma [batch_size, seq_len, hidden_dim]
        """
        # Embedding
        x = self.embedding(x)

        # LSTM
        output, _ = self.lstm(x)

        # Proietta
        output = self.linear(output)
        output = self.relu(output)

        return output


class Decoder(nn.Module):
    """
    Decoder per la sintesi vocale.
    Converte la rappresentazione latente in mel-spettrogramma.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024,
                 output_dim: int = 80, num_layers: int = 3):
        """
        Inizializza il decoder.

        Args:
            input_dim: Dimensione dell'input
            hidden_dim: Dimensione dello stato nascosto
            output_dim: Dimensione dell'output (numero di bande mel)
            num_layers: Numero di layer LSTM
        """
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input di forma [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Output di forma [batch_size, seq_len, output_dim]
        """
        # LSTM
        output, _ = self.lstm(x)

        # Proietta nell'output
        output = self.linear(output)

        return output


class VoiceModel:
    """Classe per l'addestramento e l'utilizzo del modello vocale."""

    def __init__(self, model_dir: str, debug: bool = False):
        """
        Inizializza il modello vocale.

        Args:
            model_dir: Directory per i modelli
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.voice_model")
        self.model_dir = model_dir
        self.debug = debug

        # Verifica se CUDA è disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parametri del modello
        self.input_dim = 80  # Dimensione dell'input (numero di bande mel)
        self.hidden_dim = 512  # Dimensione dello stato nascosto
        self.embedding_dim = 512  # Dimensione dell'embedding
        self.vocab_size = 1000  # Dimensione del vocabolario (sarà aggiornata durante l'addestramento)

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
        """
        Addestra un modello vocale basato sulle caratteristiche estratte.

        Args:
            features: Caratteristiche audio estratte
            model_name: Nome del modello
            model_path: Percorso del modello
            progress_callback: Funzione di callback per il progresso

        Returns:
            str: Percorso del modello salvato
        """
        self.logger.info(f"Addestramento del modello vocale: {model_name}")

        try:
            # Assicura che la directory esista
            os.makedirs(model_path, exist_ok=True)

            # Estrai le caratteristiche
            mel_spectrogram = features["features"]["mel_spectrogram"]
            mfcc = features["features"]["mfcc"]
            pitch = features["features"]["pitch"]["f0"]
            energy = features["features"]["energy"]

            # Converti in tensori PyTorch
            mel_spectrogram_tensor = torch.FloatTensor(mel_spectrogram).to(self.device)
            mfcc_tensor = torch.FloatTensor(mfcc).to(self.device)
            pitch_tensor = torch.FloatTensor(pitch).to(self.device)
            energy_tensor = torch.FloatTensor(energy).to(self.device)

            # Aggiungi la dimensione del batch
            mel_spectrogram_tensor = mel_spectrogram_tensor.unsqueeze(0)  # [1, n_mels, time]
            mfcc_tensor = mfcc_tensor.unsqueeze(0)  # [1, n_mfcc, time]
            pitch_tensor = pitch_tensor.unsqueeze(0).unsqueeze(1)  # [1, 1, time]
            energy_tensor = energy_tensor.unsqueeze(0).unsqueeze(1)  # [1, 1, time]

            # Trasponi per ottenere [batch, time, features]
            mel_spectrogram_tensor = mel_spectrogram_tensor.transpose(1, 2)  # [1, time, n_mels]
            mfcc_tensor = mfcc_tensor.transpose(1, 2)  # [1, time, n_mfcc]

            # Inizializza i modelli
            self._initialize_models()

            # Addestra il modello
            self._train_model(
                mel_spectrogram_tensor=mel_spectrogram_tensor,
                mfcc_tensor=mfcc_tensor,
                pitch_tensor=pitch_tensor,
                energy_tensor=energy_tensor,
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

            self.logger.info(f"Modello vocale addestrato e salvato: {model_file}")
            return model_file

        except Exception as e:
            self.logger.error(f"Errore durante l'addestramento del modello: {e}")
            raise

    def _initialize_models(self):
        """Inizializza i modelli."""
        # Voice Encoder
        self.voice_encoder = VoiceEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # Text Encoder
        self.text_encoder = TextEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Decoder
        self.decoder = Decoder(
            input_dim=self.hidden_dim + self.embedding_dim,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.input_dim
        ).to(self.device)

        # Ottimizzatori
        self.voice_encoder_optimizer = optim.Adam(self.voice_encoder.parameters(), lr=0.001)
        self.text_encoder_optimizer = optim.Adam(self.text_encoder.parameters(), lr=0.001)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.001)

    def _train_model(self, mel_spectrogram_tensor: torch.Tensor,
                     mfcc_tensor: torch.Tensor,
                     pitch_tensor: torch.Tensor,
                     energy_tensor: torch.Tensor,
                     num_epochs: int = 100,
                     progress_callback: Optional[Callable[[float], None]] = None):
        """
        Addestra il modello.

        Args:
            mel_spectrogram_tensor: Mel-spettrogramma
            mfcc_tensor: MFCC
            pitch_tensor: Pitch
            energy_tensor: Energia
            num_epochs: Numero di epoche
            progress_callback: Funzione di callback per il progresso
        """
        # Imposta i modelli in modalità di addestramento
        self.voice_encoder.train()
        self.text_encoder.train()
        self.decoder.train()

        # Funzione di perdita
        criterion = nn.MSELoss()

        # Addestramento
        for epoch in range(num_epochs):
            # Azzera i gradienti
            self.voice_encoder_optimizer.zero_grad()
            self.text_encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Forward pass
            # Voice Encoder
            voice_embedding = self.voice_encoder(mel_spectrogram_tensor)

            # Text Encoder (simula l'input di testo con un tensore casuale)
            batch_size = mel_spectrogram_tensor.size(0)
            seq_len = mel_spectrogram_tensor.size(1)
            text_input = torch.randint(0, self.vocab_size, (batch_size, seq_len // 2)).to(self.device)
            text_encoding = self.text_encoder(text_input)

            # Espandi l'embedding vocale per concatenarlo con l'encoding del testo
            voice_embedding_expanded = voice_embedding.unsqueeze(1).expand(-1, text_encoding.size(1), -1)

            # Concatena l'embedding vocale con l'encoding del testo
            decoder_input = torch.cat([text_encoding, voice_embedding_expanded], dim=2)

            # Decoder
            output = self.decoder(decoder_input)

            # Ridimensiona l'output per adattarlo al target
            target = mel_spectrogram_tensor[:, :output.size(1), :]

            # Calcola la perdita
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Aggiorna i pesi
            self.voice_encoder_optimizer.step()
            self.text_encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Log
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoca {epoch + 1}/{num_epochs}, Perdita: {loss.item():.4f}")

            # Aggiorna il progresso
            if progress_callback:
                progress = (epoch + 1) / num_epochs
                progress_callback(progress)

    def _save_model(self, model_file: str):
        """
        Salva il modello.

        Args:
            model_file: Percorso del file del modello
        """
        # Crea il dizionario del modello
        model_dict = {
            "voice_encoder": self.voice_encoder.state_dict(),
            "text_encoder": self.text_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size
        }

        # Salva il modello
        torch.save(model_dict, model_file)

    def load_model(self, model_file: str) -> bool:
        """
        Carica un modello.

        Args:
            model_file: Percorso del file del modello

        Returns:
            bool: True se il caricamento è riuscito, False altrimenti
        """
        try:
            # Carica il modello
            model_dict = torch.load(model_file, map_location=self.device)

            # Estrai i parametri
            self.input_dim = model_dict["input_dim"]
            self.hidden_dim = model_dict["hidden_dim"]
            self.embedding_dim = model_dict["embedding_dim"]
            self.vocab_size = model_dict["vocab_size"]

            # Inizializza i modelli
            self._initialize_models()

            # Carica i pesi
            self.voice_encoder.load_state_dict(model_dict["voice_encoder"])
            self.text_encoder.load_state_dict(model_dict["text_encoder"])
            self.decoder.load_state_dict(model_dict["decoder"])

            # Imposta i modelli in modalità di valutazione
            self.voice_encoder.eval()
            self.text_encoder.eval()
            self.decoder.eval()

            self.logger.info(f"Modello caricato: {model_file}")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante il caricamento del modello: {e}")
            return False

    def get_embedding(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Ottiene l'embedding vocale da un mel-spettrogramma.

        Args:
            mel_spectrogram: Mel-spettrogramma di forma [n_mels, time]

        Returns:
            np.ndarray: Embedding vocale di forma [embedding_dim]
        """
        try:
            # Converti in tensore e aggiungi dimensione batch
            mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0).to(self.device)

            # Trasponi per ottenere [batch, time, n_mels]
            mel_tensor = mel_tensor.transpose(1, 2)

            # Calcola l'embedding
            with torch.no_grad():
                embedding = self.voice_encoder(mel_tensor)

            # Converti in numpy
            return embedding.cpu().numpy()[0]

        except Exception as e:
            self.logger.error(f"Errore durante il calcolo dell'embedding: {e}")
            raise

    def synthesize(self, text: Union[str, List[int]], voice_embedding: np.ndarray,
                   progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Sintetizza il parlato dal testo e dall'embedding vocale.

        Args:
            text: Testo da sintetizzare (stringa o lista di token)
            voice_embedding: Embedding vocale di forma [embedding_dim]
            progress_callback: Callback per tracciare il progresso

        Returns:
            np.ndarray: Mel-spettrogramma sintetizzato di forma [n_mels, time]
        """
        try:
            # Prepara l'input di testo
            if isinstance(text, str):
                # Tokenizza il testo (implementazione semplificata)
                text_tokens = [ord(c) % self.vocab_size for c in text]
            else:
                text_tokens = text

            # Converti in tensore e aggiungi dimensione batch
            text_tensor = torch.LongTensor(text_tokens).unsqueeze(0).to(self.device)
            voice_tensor = torch.FloatTensor(voice_embedding).unsqueeze(0).to(self.device)

            if progress_callback:
                progress_callback(0.1)

            # Encoding del testo
            with torch.no_grad():
                text_encoding = self.text_encoder(text_tensor)

            if progress_callback:
                progress_callback(0.4)

            # Espandi l'embedding vocale
            voice_embedding_expanded = voice_tensor.unsqueeze(1).expand(
                -1, text_encoding.size(1), -1)

            # Concatena con l'encoding del testo
            decoder_input = torch.cat([text_encoding, voice_embedding_expanded], dim=2)

            if progress_callback:
                progress_callback(0.6)

            # Decoding
            with torch.no_grad():
                mel_output = self.decoder(decoder_input)

            # Converti in numpy e rimuovi la dimensione batch
            mel_output = mel_output.squeeze(0).cpu().numpy()

            # Trasponi per ottenere [n_mels, time]
            mel_output = mel_output.T

            if progress_callback:
                progress_callback(1.0)

            return mel_output

        except Exception as e:
            self.logger.error(f"Errore durante la sintesi vocale: {e}")
            raise

    def save_vocabulary(self, vocab_file: str) -> bool:
        """
        Salva il vocabolario su file.

        Args:
            vocab_file: Percorso del file del vocabolario

        Returns:
            bool: True se il salvataggio è riuscito, False altrimenti
        """
        try:
            if self.vocab is None:
                self.logger.warning("Nessun vocabolario da salvare")
                return False

            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Vocabolario salvato: {vocab_file}")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante il salvataggio del vocabolario: {e}")
            return False

    def load_vocabulary(self, vocab_file: str) -> bool:
        """
        Carica il vocabolario da file.

        Args:
            vocab_file: Percorso del file del vocabolario

        Returns:
            bool: True se il caricamento è riuscito, False altrimenti
        """
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

            # Aggiorna la dimensione del vocabolario
            self.vocab_size = len(self.vocab)

            # Ricrea il text encoder con la nuova dimensione del vocabolario
            if self.text_encoder is not None:
                self.text_encoder = TextEncoder(
                    vocab_size=self.vocab_size,
                    embedding_dim=self.embedding_dim,
                    hidden_dim=self.hidden_dim
                ).to(self.device)
                self.text_encoder_optimizer = optim.Adam(self.text_encoder.parameters(), lr=0.001)

            self.logger.info(f"Vocabolario caricato: {vocab_file} (size={self.vocab_size})")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante il caricamento del vocabolario: {e}")
            return False

    def cleanup(self):
        """Libera le risorse e la memoria."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.voice_encoder = None
        self.text_encoder = None
        self.decoder = None