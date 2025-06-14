import os
import sys
import tempfile
import logging
import numpy as np
import torch

# Aggiungi il percorso del progetto al PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_cloning import VoiceCloner
from voice_model_trainer import VoiceModel
from controller import Controller

def test_voice_cloning_save():
    print("Test del salvataggio del modello di clonazione vocale...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cloner = VoiceCloner(model_dir=temp_dir, debug=True)

        audio_data = {
            "features": {
                "mel_spectrogram": np.random.rand(80, 100),
                "mfcc": np.random.rand(13, 100),
                "pitch": {"f0": np.random.rand(100)},
                "energy": np.random.rand(100)
            }
        }

        model_name = "test_model"
        model_path = cloner.clone_voice(audio_data, model_name)

        assert os.path.exists(model_path), f"Modello non salvato: {model_path}"
        assert os.path.getsize(model_path) > 0, f"File del modello vuoto: {model_path}"

        metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
        assert os.path.exists(metadata_path), f"Metadati non salvati: {metadata_path}"

        print(f"✓ Modello di clonazione vocale salvato correttamente: {model_path} ({os.path.getsize(model_path)} bytes)")
        assert True

def test_voice_model_trainer_save():
    print("Test del salvataggio del modello di addestramento vocale...")

    with tempfile.TemporaryDirectory() as temp_dir:
        voice_model = VoiceModel(model_dir=temp_dir, debug=True)

        features = {
            "features": {
                "mel_spectrogram": np.random.rand(80, 100),
                "mfcc": np.random.rand(13, 100),
                "pitch": {"f0": np.random.rand(100)},
                "energy": np.random.rand(100)
            }
        }

        model_name = "test_trainer_model"
        model_path = os.path.join(temp_dir, model_name)

        saved_model_path = voice_model.train(
            features=features,
            model_name=model_name,
            model_path=model_path
        )

        assert os.path.exists(saved_model_path), f"Modello non salvato: {saved_model_path}"
        assert os.path.getsize(saved_model_path) > 0, f"File del modello vuoto: {saved_model_path}"

        metadata_path = os.path.join(model_path, "metadata.json")
        assert os.path.exists(metadata_path), f"Metadati non salvati: {metadata_path}"

        print(f"✓ Modello di addestramento vocale salvato correttamente: {saved_model_path} ({os.path.getsize(saved_model_path)} bytes)")
        assert True

def test_model_loading():
    print("Test del caricamento del modello...")

    with tempfile.TemporaryDirectory() as temp_dir:
        cloner = VoiceCloner(model_dir=temp_dir, debug=True)

        audio_data = {
            "features": {
                "mel_spectrogram": np.random.rand(80, 100),
                "mfcc": np.random.rand(13, 100),
                "pitch": {"f0": np.random.rand(100)},
                "energy": np.random.rand(100)
            }
        }

        model_name = "test_load_model"
        model_path = cloner.clone_voice(audio_data, model_name)

        cloner2 = VoiceCloner(model_dir=temp_dir, debug=True)
        cloner2._load_model(model_path)

        print(f"✓ Modello caricato correttamente: {model_path}")
        assert True

def test_controller_get_available_models():
    print("Test del metodo get_available_models del controller...")

    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        controller = Controller(model_dir=models_dir, debug=True)

        assert len(controller.get_available_models()) == 0, "Dovrebbero esserci 0 modelli inizialmente"

        # Modello valido
        model_dir = os.path.join(models_dir, "test_model")
        os.makedirs(model_dir, exist_ok=True)
        torch.save({"test": "data"}, os.path.join(model_dir, "model.pt"))
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            f.write('{"model_name": "test_model"}')

        # Modello incompleto (senza metadata.json)
        incomplete_model_dir = os.path.join(models_dir, "incomplete_model")
        os.makedirs(incomplete_model_dir, exist_ok=True)
        torch.save({"test": "data"}, os.path.join(incomplete_model_dir, "model.pt"))

        models = controller.get_available_models()

        assert "test_model" in models, "Il modello valido non è stato rilevato"
        assert "incomplete_model" not in models, "Il modello incompleto non dovrebbe essere rilevato"
        assert len(models) == 1, f"Dovrebbe esserci solo un modello valido, trovati {len(models)}"

        print("✓ Metodo get_available_models funziona correttamente")
        assert True
