#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test per verificare le correzioni del salvataggio del modello.
"""

import os
import sys
import tempfile
import shutil
import logging
import numpy as np
import torch

# Aggiungi il percorso del progetto al PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_cloning import VoiceCloner
from voice_model_trainer import VoiceModel
from controller import Controller

def test_voice_cloning_save():
    """Test del salvataggio del modello di clonazione vocale."""
    print("Test del salvataggio del modello di clonazione vocale...")

    # Crea una directory temporanea
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Inizializza il clonatore vocale
            cloner = VoiceCloner(model_dir=temp_dir, debug=True)

            # Crea dati audio fittizi
            audio_data = {
                "features": {
                    "mel_spectrogram": np.random.rand(80, 100),
                    "mfcc": np.random.rand(13, 100),
                    "pitch": {"f0": np.random.rand(100)},
                    "energy": np.random.rand(100)
                }
            }

            # Clona la voce
            model_name = "test_model"
            model_path = cloner.clone_voice(audio_data, model_name)

            # Verifica che il modello sia stato salvato
            assert os.path.exists(model_path), f"Modello non salvato: {model_path}"

            # Verifica che il file non sia vuoto
            file_size = os.path.getsize(model_path)
            assert file_size > 0, f"File del modello vuoto: {model_path}"

            # Verifica che i metadati siano stati salvati
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            assert os.path.exists(metadata_path), f"Metadati non salvati: {metadata_path}"

            print(f"✓ Modello di clonazione vocale salvato correttamente: {model_path} ({file_size} bytes)")
            return True

        except Exception as e:
            print(f"✗ Errore nel test del salvataggio del modello di clonazione vocale: {e}")
            return False


def test_voice_model_trainer_save():
    """Test del salvataggio del modello di addestramento vocale."""
    print("Test del salvataggio del modello di addestramento vocale...")

    # Crea una directory temporanea
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Inizializza il modello vocale
            voice_model = VoiceModel(model_dir=temp_dir, debug=True)

            # Crea dati audio fittizi
            features = {
                "features": {
                    "mel_spectrogram": np.random.rand(80, 100),
                    "mfcc": np.random.rand(13, 100),
                    "pitch": {"f0": np.random.rand(100)},
                    "energy": np.random.rand(100)
                }
            }

            # Addestra il modello
            model_name = "test_trainer_model"
            model_path = os.path.join(temp_dir, model_name)

            saved_model_path = voice_model.train(
                features=features,
                model_name=model_name,
                model_path=model_path
            )

            # Verifica che il modello sia stato salvato
            assert os.path.exists(saved_model_path), f"Modello non salvato: {saved_model_path}"

            # Verifica che il file non sia vuoto
            file_size = os.path.getsize(saved_model_path)
            assert file_size > 0, f"File del modello vuoto: {saved_model_path}"

            # Verifica che i metadati siano stati salvati
            metadata_path = os.path.join(model_path, "metadata.json")
            assert os.path.exists(metadata_path), f"Metadati non salvati: {metadata_path}"

            print(f"✓ Modello di addestramento vocale salvato correttamente: {saved_model_path} ({file_size} bytes)")
            return True

        except Exception as e:
            print(f"✗ Errore nel test del salvataggio del modello di addestramento vocale: {e}")
            return False


def test_model_loading():
    """Test del caricamento del modello."""
    print("Test del caricamento del modello...")

    # Crea una directory temporanea
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Inizializza il clonatore vocale
            cloner = VoiceCloner(model_dir=temp_dir, debug=True)

            # Crea dati audio fittizi
            audio_data = {
                "features": {
                    "mel_spectrogram": np.random.rand(80, 100),
                    "mfcc": np.random.rand(13, 100),
                    "pitch": {"f0": np.random.rand(100)},
                    "energy": np.random.rand(100)
                }
            }

            # Clona la voce
            model_name = "test_load_model"
            model_path = cloner.clone_voice(audio_data, model_name)

            # Crea un nuovo clonatore e carica il modello
            cloner2 = VoiceCloner(model_dir=temp_dir, debug=True)
            cloner2._load_model(model_path)

            print(f"✓ Modello caricato correttamente: {model_path}")
            return True

        except Exception as e:
            print(f"✗ Errore nel test del caricamento del modello: {e}")
            return False


def test_controller_get_available_models():
    """Test del metodo get_available_models del controller."""
    print("Test del metodo get_available_models del controller...")

    # Crea una directory temporanea
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Crea directory dei modelli
            models_dir = os.path.join(temp_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            # Inizializza il controller
            controller = Controller(model_dir=models_dir, debug=True)

            # Verifica che non ci siano modelli inizialmente
            models = controller.get_available_models()
            assert len(models) == 0, f"Dovrebbero esserci 0 modelli, trovati {len(models)}"

            # Crea un modello fittizio
            model_dir = os.path.join(models_dir, "test_model")
            os.makedirs(model_dir, exist_ok=True)

            # Crea il file del modello
            model_file = os.path.join(model_dir, "model.pt")
            torch.save({"test": "data"}, model_file)

            # Crea i metadati
            metadata_file = os.path.join(model_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                f.write('{"model_name": "test_model"}')

            # Verifica che il modello sia rilevato
            models = controller.get_available_models()
            assert len(models) == 1, f"Dovrebbe esserci 1 modello, trovati {len(models)}"
            assert "test_model" in models, f"Il modello test_model non è stato trovato"

            # Crea un modello incompleto (senza metadati)
            incomplete_model_dir = os.path.join(models_dir, "incomplete_model")
            os.makedirs(incomplete_model_dir, exist_ok=True)
            incomplete_model_file = os.path.join(incomplete_model_dir, "model.pt")
            torch.save({"test": "data"}, incomplete_model_file)

            # Verifica che il modello incompleto non sia rilevato
            models = controller.get_available_models()
            assert len(models) == 1, f"Dovrebbe esserci ancora 1 modello, trovati {len(models)}"
            assert "incomplete_model" not in models, f"Il modello incompleto non dovrebbe essere rilevato"

            print(f"✓ Metodo get_available_models funziona correttamente")
            return True

        except Exception as e:
            print(f"✗ Errore nel test del metodo get_available_models: {e}")
            return False


def main():
    """Funzione principale per eseguire tutti i test."""
    print("Avvio dei test per le correzioni del salvataggio del modello...")
    print("=" * 60)

    # Configura il logging
    logging.basicConfig(level=logging.WARNING)

    # Esegui i test
    tests = [
        test_voice_cloning_save,
        test_voice_model_trainer_save,
        test_model_loading,
        test_controller_get_available_models
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Errore durante l'esecuzione del test {test.__name__}: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Risultati dei test:")
    print(f"✓ Test superati: {passed}")
    print(f"✗ Test falliti: {failed}")
    print(f"Totale: {passed + failed}")

    if failed == 0:
        print("\n🎉 Tutti i test sono stati superati! Le correzioni funzionano correttamente.")
        return 0
    else:
        print(f"\n❌ {failed} test sono falliti. Verificare le correzioni.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

