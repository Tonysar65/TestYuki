#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test per verificare che i file creati siano compatibili con il repository TestYuki.
"""

import os
import json
import pickle
import unittest


class TestTestYukiFiles(unittest.TestCase):
    """Test per verificare la compatibilità dei file con TestYuki."""

    def setUp(self):
        """Configurazione per ogni test."""
        self.base_dir = "voice_models/example_model"
        self.model_path = os.path.join(self.base_dir, "model.pt")
        self.metadata_path = os.path.join(self.base_dir, "metadata.json")

    def test_files_exist(self):
        """Test che verifica l'esistenza dei file."""
        self.assertTrue(os.path.exists(self.model_path), "Il file model.pt deve esistere")
        self.assertTrue(os.path.exists(self.metadata_path), "Il file metadata.json deve esistere")

    def test_files_not_empty(self):
        """Test che verifica che i file non siano vuoti."""
        model_size = os.path.getsize(self.model_path)
        metadata_size = os.path.getsize(self.metadata_path)

        self.assertGreater(model_size, 0, "Il file model.pt non deve essere vuoto")
        self.assertGreater(metadata_size, 0, "Il file metadata.json non deve essere vuoto")

        print(f"✅ model.pt: {model_size:,} bytes")
        print(f"✅ metadata.json: {metadata_size:,} bytes")

    def test_metadata_structure(self):
        """Test che verifica la struttura del file metadata.json."""
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Verifica le chiavi necessarie per TestYuki
        required_keys = [
            "model_name", "input_dim", "hidden_dim", "embedding_dim",
            "vocab_size", "created_at", "device"
        ]

        for key in required_keys:
            self.assertIn(key, metadata, f"Chiave mancante nei metadati: {key}")

        # Verifica i tipi di dati
        self.assertIsInstance(metadata["model_name"], str)
        self.assertIsInstance(metadata["input_dim"], int)
        self.assertIsInstance(metadata["hidden_dim"], int)
        self.assertIsInstance(metadata["embedding_dim"], int)
        self.assertIsInstance(metadata["vocab_size"], int)
        self.assertIsInstance(metadata["created_at"], (int, float))
        self.assertIsInstance(metadata["device"], str)

        # Verifica i valori
        self.assertEqual(metadata["input_dim"], 80)
        self.assertEqual(metadata["hidden_dim"], 512)
        self.assertEqual(metadata["embedding_dim"], 512)
        self.assertEqual(metadata["vocab_size"], 1000)

        print("✅ Struttura metadata.json verificata")

    def test_model_structure(self):
        """Test che verifica la struttura del file model.pt."""
        with open(self.model_path, 'rb') as f:
            model_dict = pickle.load(f)

        # Verifica le chiavi necessarie per TestYuki
        required_keys = [
            "voice_encoder", "text_encoder", "decoder",
            "input_dim", "hidden_dim", "embedding_dim", "vocab_size"
        ]

        for key in required_keys:
            self.assertIn(key, model_dict, f"Chiave mancante nel modello: {key}")

        # Verifica che i moduli abbiano i parametri necessari
        voice_encoder = model_dict["voice_encoder"]
        text_encoder = model_dict["text_encoder"]
        decoder = model_dict["decoder"]

        # Verifica VoiceEncoder
        self.assertIn("lstm.weight_ih_l0", voice_encoder)
        self.assertIn("projection.weight", voice_encoder)

        # Verifica TextEncoder
        self.assertIn("embedding.weight", text_encoder)
        self.assertIn("lstm.weight_ih_l0", text_encoder)

        # Verifica Decoder
        self.assertIn("lstm.weight_ih_l0", decoder)
        self.assertIn("projection.weight", decoder)

        print("✅ Struttura model.pt verificata")

    def test_testyuki_compatibility(self):
        """Test che simula il caricamento come farebbe TestYuki."""

        # Simula il metodo _verify_model_exists del controller corretto
        def _verify_model_exists(model_path):
            try:
                if not os.path.exists(model_path):
                    return False

                model_file = os.path.join(model_path, "model.pt")
                if not os.path.exists(model_file):
                    return False

                metadata_file = os.path.join(model_path, "metadata.json")
                if not os.path.exists(metadata_file):
                    return False

                if os.path.getsize(model_file) == 0:
                    return False

                if os.path.getsize(metadata_file) == 0:
                    return False

                return True
            except Exception:
                return False

        # Test
        result = _verify_model_exists(self.base_dir)
        self.assertTrue(result, "Il modello deve essere riconosciuto come valido da TestYuki")

        # Simula il caricamento del modello
        def simulate_load_model(model_file):
            try:
                with open(model_file, 'rb') as f:
                    model_dict = pickle.load(f)

                # Verifica che le chiavi necessarie siano presenti
                required_keys = ["voice_encoder", "text_encoder", "decoder", "input_dim", "hidden_dim"]
                for key in required_keys:
                    if key not in model_dict:
                        return False

                return True
            except Exception:
                return False

        load_result = simulate_load_model(self.model_path)
        self.assertTrue(load_result, "Il modello deve poter essere caricato da TestYuki")

        print("✅ Compatibilità con TestYuki verificata")

    def test_get_available_models_filtering(self):
        """Test che simula il filtro dei modelli disponibili di TestYuki."""

        def get_available_models(model_dir):
            models = []
            try:
                if not os.path.exists(model_dir):
                    return models

                for item in os.listdir(model_dir):
                    model_path = os.path.join(model_dir, item)
                    if os.path.isdir(model_path):
                        # Verifica che il modello sia valido
                        model_file = os.path.join(model_path, "model.pt")
                        metadata_file = os.path.join(model_path, "metadata.json")

                        if (os.path.exists(model_file) and
                                os.path.exists(metadata_file) and
                                os.path.getsize(model_file) > 0 and
                                os.path.getsize(metadata_file) > 0):
                            models.append(item)

                return sorted(models)
            except Exception:
                return []

        models = get_available_models(os.path.dirname(self.base_dir))
        self.assertIn("example_model", models, "Il modello deve essere incluso nell'elenco dei modelli disponibili")

        print(f"✅ Modelli disponibili: {models}")


def run_compatibility_tests():
    """Esegue tutti i test di compatibilità."""
    print("=== TEST DI COMPATIBILITÀ CON TESTYUKI ===")
    print()

    # Crea la suite di test
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTestYukiFiles)

    # Esegue i test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=== RIEPILOGO DEI TEST ===")
    print(f"Test eseguiti: {result.testsRun}")
    print(f"Errori: {len(result.errors)}")
    print(f"Fallimenti: {len(result.failures)}")

    if result.errors:
        print("\nERRORI:")
        for test, error in result.errors:
            print(f"- {test}: {error}")

    if result.failures:
        print("\nFALLIMENTI:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")

    if result.wasSuccessful():
        print("\n✅ TUTTI I TEST SONO STATI SUPERATI!")
        print("I file creati sono compatibili con il repository TestYuki.")
    else:
        print("\n❌ ALCUNI TEST SONO FALLITI!")
        print("Potrebbero essere necessarie correzioni ai file.")

    return result.wasSuccessful()


def create_installation_instructions():
    """Crea le istruzioni per l'installazione dei file."""
    instructions = """
# ISTRUZIONI PER L'INSTALLAZIONE DEI FILE

## 1. Copia i file nel repository TestYuki

Copia i seguenti file nella directory del tuo repository TestYuki:

```bash
# Crea la directory del modello se non esiste
mkdir -p /path/to/TestYuki/voice_models/example_model

# Copia i file
cp /home/ubuntu/testyuki_fix/voice_models/example_model/model.pt /path/to/TestYuki/voice_models/example_model/
cp /home/ubuntu/testyuki_fix/voice_models/example_model/metadata.json /path/to/TestYuki/voice_models/example_model/
```

## 2. Verifica l'installazione

Esegui il seguente comando per verificare che i file siano stati copiati correttamente:

```bash
ls -la /path/to/TestYuki/voice_models/example_model/
```

Dovresti vedere:
- model.pt (circa 30 MB)
- metadata.json (circa 1 KB)

## 3. Test del modello

Ora puoi testare il modello nel tuo repository TestYuki:

1. Avvia l'applicazione TestYuki
2. Il modello "example_model" dovrebbe apparire nell'elenco dei modelli disponibili
3. Prova a utilizzarlo per la sintesi vocale

## 4. Risoluzione dei problemi

Se il modello non appare nell'elenco:
1. Verifica che i file siano nella posizione corretta
2. Verifica che i file non siano vuoti
3. Controlla i log dell'applicazione per eventuali errori

## 5. Creazione di nuovi modelli

Per creare nuovi modelli:
1. Usa la struttura di directory: voice_models/nome_modello/
2. Assicurati che ogni modello abbia sia model.pt che metadata.json
3. Usa i file di esempio come template per la struttura corretta
"""

    with open("docs/ISTRUZIONI_INSTALLAZIONE.md", "w") as f:
        f.write(instructions)

    print("✅ Istruzioni di installazione create: docs/ISTRUZIONI_INSTALLAZIONE.md")


if __name__ == "__main__":
    # Esegue i test di compatibilità
    success = run_compatibility_tests()

    if success:
        print()
        print("=== CREAZIONE ISTRUZIONI DI INSTALLAZIONE ===")
        create_installation_instructions()

        print()
        print("🎉 TUTTO COMPLETATO CON SUCCESSO!")
        print("I file sono pronti per essere utilizzati con TestYuki.")
    else:
        print()
        print("❌ ERRORE: I test di compatibilità sono falliti.")
        print("I file potrebbero non essere compatibili con TestYuki.")

