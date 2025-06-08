#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script per creare un file model.pt simulato senza PyTorch.
"""

import os
import json
import pickle
import struct
import hashlib


def create_simulated_pytorch_model(model_path):
    """
    Crea un file model.pt simulato con una struttura realistica.

    Args:
        model_path: Percorso dove salvare il file model.pt
    """
    print("Creazione del file model.pt simulato...")

    # Parametri del modello (devono corrispondere a metadata.json)
    input_dim = 80
    hidden_dim = 512
    embedding_dim = 512
    vocab_size = 1000

    # Simula i state_dict dei modelli con tensori fittizi
    def create_fake_tensor_data(shape):
        """Crea dati fittizi per simulare un tensore."""
        import random
        size = 1
        for dim in shape:
            size *= dim
        # Crea dati float32 casuali
        data = []
        for _ in range(size):
            data.append(struct.pack('f', random.uniform(-1.0, 1.0)))
        return b''.join(data)

    # Struttura simulata del modello PyTorch
    model_dict = {
        "voice_encoder": {
            "lstm.weight_ih_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4, input_dim]),
                "shape": [hidden_dim * 4, input_dim],
                "dtype": "float32"
            },
            "lstm.weight_hh_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4, hidden_dim]),
                "shape": [hidden_dim * 4, hidden_dim],
                "dtype": "float32"
            },
            "lstm.bias_ih_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4]),
                "shape": [hidden_dim * 4],
                "dtype": "float32"
            },
            "lstm.bias_hh_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4]),
                "shape": [hidden_dim * 4],
                "dtype": "float32"
            },
            "projection.weight": {
                "data": create_fake_tensor_data([embedding_dim, hidden_dim * 2]),
                "shape": [embedding_dim, hidden_dim * 2],
                "dtype": "float32"
            },
            "projection.bias": {
                "data": create_fake_tensor_data([embedding_dim]),
                "shape": [embedding_dim],
                "dtype": "float32"
            }
        },
        "text_encoder": {
            "embedding.weight": {
                "data": create_fake_tensor_data([vocab_size, embedding_dim]),
                "shape": [vocab_size, embedding_dim],
                "dtype": "float32"
            },
            "lstm.weight_ih_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4, embedding_dim]),
                "shape": [hidden_dim * 4, embedding_dim],
                "dtype": "float32"
            },
            "lstm.weight_hh_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4, hidden_dim]),
                "shape": [hidden_dim * 4, hidden_dim],
                "dtype": "float32"
            },
            "lstm.bias_ih_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4]),
                "shape": [hidden_dim * 4],
                "dtype": "float32"
            },
            "lstm.bias_hh_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4]),
                "shape": [hidden_dim * 4],
                "dtype": "float32"
            }
        },
        "decoder": {
            "lstm.weight_ih_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4, hidden_dim * 2]),
                "shape": [hidden_dim * 4, hidden_dim * 2],
                "dtype": "float32"
            },
            "lstm.weight_hh_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4, hidden_dim]),
                "shape": [hidden_dim * 4, hidden_dim],
                "dtype": "float32"
            },
            "lstm.bias_ih_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4]),
                "shape": [hidden_dim * 4],
                "dtype": "float32"
            },
            "lstm.bias_hh_l0": {
                "data": create_fake_tensor_data([hidden_dim * 4]),
                "shape": [hidden_dim * 4],
                "dtype": "float32"
            },
            "projection.weight": {
                "data": create_fake_tensor_data([input_dim, hidden_dim]),
                "shape": [input_dim, hidden_dim],
                "dtype": "float32"
            },
            "projection.bias": {
                "data": create_fake_tensor_data([input_dim]),
                "shape": [input_dim],
                "dtype": "float32"
            }
        },
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "vocab_size": vocab_size,
        "model_info": {
            "created_with": "testyuki_fix_script_simulated",
            "architecture": "voice_cloning_model",
            "format": "simulated_pytorch"
        }
    }

    # Salva il modello usando pickle (simula il formato PyTorch)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Verifica che il file sia stato creato
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"✅ File model.pt simulato creato con successo!")
            print(f"   Percorso: {model_path}")
            print(f"   Dimensione: {file_size:,} bytes")

            # Verifica che il file possa essere caricato
            try:
                with open(model_path, 'rb') as f:
                    loaded_dict = pickle.load(f)

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

    except Exception as e:
        print(f"❌ Errore durante la creazione del modello: {e}")
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

            # Calcola un checksum MD5
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            metadata["file_info"]["checksum"] = file_hash

            # Aggiorna il timestamp
            import time
            metadata["created_at"] = time.time()
            metadata["file_info"]["format"] = "simulated_pytorch"

        # Salva i metadati aggiornati
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadati aggiornati con le informazioni del file model.pt")
        return True

    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento dei metadati: {e}")
        return False


def verify_files(base_dir):
    """
    Verifica che entrambi i file siano stati creati correttamente.

    Args:
        base_dir: Directory base contenente i file
    """
    model_path = os.path.join(base_dir, "model.pt")
    metadata_path = os.path.join(base_dir, "metadata.json")

    print("=== VERIFICA FINALE ===")

    results = {
        "model_exists": os.path.exists(model_path),
        "metadata_exists": os.path.exists(metadata_path),
        "model_size": 0,
        "metadata_size": 0,
        "model_valid": False,
        "metadata_valid": False
    }

    if results["model_exists"]:
        results["model_size"] = os.path.getsize(model_path)
        print(f"✅ model.pt: {results['model_size']:,} bytes")

        # Verifica che il file possa essere caricato
        try:
            with open(model_path, 'rb') as f:
                loaded_dict = pickle.load(f)
            required_keys = ["voice_encoder", "text_encoder", "decoder"]
            if all(key in loaded_dict for key in required_keys):
                results["model_valid"] = True
                print("✅ model.pt: struttura valida")
            else:
                print("⚠️  model.pt: struttura non valida")
        except:
            print("❌ model.pt: impossibile caricare il file")
    else:
        print("❌ model.pt: file non trovato")

    if results["metadata_exists"]:
        results["metadata_size"] = os.path.getsize(metadata_path)
        print(f"✅ metadata.json: {results['metadata_size']:,} bytes")

        # Verifica che il JSON sia valido
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            required_keys = ["model_name", "input_dim", "hidden_dim"]
            if all(key in metadata for key in required_keys):
                results["metadata_valid"] = True
                print("✅ metadata.json: struttura valida")
            else:
                print("⚠️  metadata.json: struttura non valida")
        except:
            print("❌ metadata.json: JSON non valido")
    else:
        print("❌ metadata.json: file non trovato")

    # Risultato finale
    success = (results["model_exists"] and results["metadata_exists"] and
               results["model_size"] > 0 and results["metadata_size"] > 0 and
               results["model_valid"] and results["metadata_valid"])

    print()
    if success:
        print("🎉 SUCCESSO! Entrambi i file sono stati creati correttamente e sono validi.")
    else:
        print("❌ ERRORE: Uno o più problemi rilevati nei file.")

    return success, results


if __name__ == "__main__":
    import random

    random.seed(42)  # Per risultati riproducibili

    # Percorsi dei file
    base_dir = "voice_models/example_model"
    model_path = os.path.join(base_dir, "model.pt")
    metadata_path = os.path.join(base_dir, "metadata.json")

    print("=== CREAZIONE FILE MODEL.PT SIMULATO PER TESTYUKI ===")
    print()

    # Crea il file model.pt simulato
    success = create_simulated_pytorch_model(model_path)

    if success:
        print()
        print("=== AGGIORNAMENTO METADATI ===")
        update_metadata_with_file_info(metadata_path, model_path)

        print()
        # Verifica finale
        verify_files(base_dir)
    else:
        print()
        print("❌ ERRORE: Impossibile creare il file model.pt")

