"""
Script per creare un file model.pt valido per il repository TestYuki.
"""

import warnings
import torch
import torch.nn as nn
import os
import json

warnings.filterwarnings("ignore")

class VoiceEncoder(nn.Module):
    """Encoder per le caratteristiche vocali."""

    def __init__(self, input_dim=80, hidden_dim=512, embedding_dim=512):
        super(VoiceEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_dim * 2]
        output = self.projection(lstm_out)
        output = self.dropout(output)
        return output


class TextEncoder(nn.Module):
    """Encoder per il testo."""

    def __init__(self, vocab_size=1000, embedding_dim=512, hidden_dim=512):
        super(TextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out


class Decoder(nn.Module):
    """Decoder per la generazione dell'audio."""

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=80):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        output = self.projection(lstm_out)
        output = self.dropout(output)
        return output


def create_model_pt(model_path):
    """
    Crea un file model.pt valido con la struttura corretta.

    Args:
        model_path: Percorso dove salvare il file model.pt
    """
    print("Creazione del file model.pt...")

    # Parametri del modello (devono corrispondere a metadata.json)
    input_dim = 80
    hidden_dim = 512
    embedding_dim = 512
    vocab_size = 1000

    # Crea i modelli
    voice_encoder = VoiceEncoder(input_dim, hidden_dim, embedding_dim)
    text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)
    decoder = Decoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=input_dim)

    # Inizializza i pesi con valori casuali ma realistici
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.1)

    voice_encoder.apply(init_weights)
    text_encoder.apply(init_weights)
    decoder.apply(init_weights)

    # Crea il dizionario del modello (deve corrispondere al formato atteso dal codice)
    model_dict = {
        "voice_encoder": voice_encoder.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "vocab_size": vocab_size,
        "model_info": {
            "pytorch_version": torch.__version__,
            "created_with": "testyuki_fix_script",
            "architecture": "voice_cloning_model"
        }
    }

    # Salva il modello
    torch.save(model_dict, model_path)

    # Verifica che il file sia stato creato
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"✅ File model.pt creato con successo!")
        print(f"   Percorso: {model_path}")
        print(f"   Dimensione: {file_size:,} bytes")

        # Verifica che il file possa essere caricato
        try:
            loaded_dict = torch.load(model_path, map_location='cpu')
            required_keys = ["voice_encoder", "text_encoder", "decoder", "input_dim", "hidden_dim"]

            missing_keys = [key for key in required_keys if key not in loaded_dict]
            if missing_keys:
                print(f"⚠️  Chiavi mancanti: {missing_keys}")
                return False
            else:
                print("✅ Struttura del modello verificata - tutte le chiavi necessarie sono presenti")
                return True

        except Exception as e:
            print(f"❌ Errore durante la verifica del modello: {e}")
            return False
    else:
        print("❌ Errore: il file model.pt non è stato creato")
        return False


def update_metadata_with_file_info(metadata_path, model_path):
    """
    Aggiorna il file metadata.json con le informazioni del file model.pt.

    Args:
        metadata_path: Percorso del file metadata.json
        model_path: Percorso del file model.pt
    """
    try:
        # Carica i metadati esistenti
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Aggiorna le informazioni del file
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            metadata["file_info"]["file_size_bytes"] = file_size

            # Calcola un checksum semplice (per esempio)
            import hashlib
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            metadata["file_info"]["checksum"] = file_hash

        # Salva i metadati aggiornati
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadati aggiornati con le informazioni del file model.pt")
        return True

    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento dei metadati: {e}")
        return False


if __name__ == "__main__":
    # Percorsi dei file
    base_dir = "./models"
    model_path = os.path.join(base_dir, "model.pt")
    metadata_path = os.path.join(base_dir, "metadata.json")

    print("=== CREAZIONE FILE MODEL.PT PER TESTYUKI ===")
    print()

    # Crea il file model.pt
    success = create_model_pt(model_path)

    if success:
        print()
        print("=== AGGIORNAMENTO METADATI ===")
        update_metadata_with_file_info(metadata_path, model_path)

        print()
        print("=== VERIFICA FINALE ===")

        # Verifica finale
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            model_size = os.path.getsize(model_path)
            metadata_size = os.path.getsize(metadata_path)

            print(f"✅ model.pt: {model_size:,} bytes")
            print(f"✅ metadata.json: {metadata_size:,} bytes")

            if model_size > 0 and metadata_size > 0:
                print()
                print("🎉 SUCCESSO! Entrambi i file sono stati creati correttamente e non sono vuoti.")
            else:
                print()
                print("❌ ERRORE: Uno o entrambi i file sono vuoti.")
        else:
            print("❌ ERRORE: Uno o entrambi i file non esistono.")
    else:
        print()
        print("❌ ERRORE: Impossibile creare il file model.pt")

