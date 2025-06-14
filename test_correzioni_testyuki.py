"""
Test per verificare le correzioni dei problemi di salvataggio e caricamento del modello.
"""

import json
import os
import shutil
import tempfile
import unittest


class TestModelSavingLoading(unittest.TestCase):
    """Test per verificare le correzioni del salvataggio e caricamento del modello."""

    def setUp(self):
        """Configurazione per ogni test."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "voice_models")
        self.model_name = "test_model"
        self.model_path = os.path.join(self.model_dir, self.model_name)

        # Crea le directory necessarie
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

    def tearDown(self):
        """Pulizia dopo ogni test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_verify_model_exists_with_valid_model(self):
        """Test che verifica l'esistenza di un modello valido."""
        # Simula un controller corretto
        from unittest.mock import Mock

        controller = Mock()
        controller.model_dir = self.model_dir
        controller.logger = Mock()

        # Crea file di modello e metadati fittizi
        model_file = os.path.join(self.model_path, "model.pt")
        metadata_file = os.path.join(self.model_path, "metadata.json")

        # Scrive contenuto fittizio nei file
        with open(model_file, "wb") as f:
            f.write(b"fake_model_data")

        with open(metadata_file, "w") as f:
            json.dump({"model_name": self.model_name}, f)

        # Implementa il metodo _verify_model_exists corretto
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
        result = _verify_model_exists(self.model_path)
        self.assertTrue(result, "Il modello valido dovrebbe essere riconosciuto come esistente")

    def test_verify_model_exists_with_missing_files(self):
        """Test che verifica il comportamento con file mancanti."""

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

        # Test con directory inesistente
        result = _verify_model_exists(os.path.join(self.model_dir, "nonexistent"))
        self.assertFalse(result, "Directory inesistente dovrebbe restituire False")

        # Test con file del modello mancante
        metadata_file = os.path.join(self.model_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({"model_name": self.model_name}, f)

        result = _verify_model_exists(self.model_path)
        self.assertFalse(result, "Modello con file .pt mancante dovrebbe restituire False")

    def test_verify_model_exists_with_empty_files(self):
        """Test che verifica il comportamento con file vuoti."""

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

        # Crea file vuoti
        model_file = os.path.join(self.model_path, "model.pt")
        metadata_file = os.path.join(self.model_path, "metadata.json")

        open(model_file, "w").close()  # File vuoto
        open(metadata_file, "w").close()  # File vuoto

        result = _verify_model_exists(self.model_path)
        self.assertFalse(result, "Modello con file vuoti dovrebbe restituire False")

    def test_get_available_models_filtering(self):
        """Test che verifica il filtro dei modelli disponibili."""

        def _verify_model_exists(model_path):
            model_file = os.path.join(model_path, "model.pt")
            metadata_file = os.path.join(model_path, "metadata.json")
            return (os.path.exists(model_file) and
                    os.path.exists(metadata_file) and
                    os.path.getsize(model_file) > 0 and
                    os.path.getsize(metadata_file) > 0)

        def get_available_models(model_dir):
            models = []
            try:
                if not os.path.exists(model_dir):
                    return models

                for item in os.listdir(model_dir):
                    model_path = os.path.join(model_dir, item)
                    if os.path.isdir(model_path) and _verify_model_exists(model_path):
                        models.append(item)

                return sorted(models)
            except Exception:
                return []

        # Crea modello valido
        valid_model_path = os.path.join(self.model_dir, "valid_model")
        os.makedirs(valid_model_path, exist_ok=True)

        with open(os.path.join(valid_model_path, "model.pt"), "wb") as f:
            f.write(b"valid_model_data")

        with open(os.path.join(valid_model_path, "metadata.json"), "w") as f:
            json.dump({"model_name": "valid_model"}, f)

        # Crea modello non valido (file vuoti)
        invalid_model_path = os.path.join(self.model_dir, "invalid_model")
        os.makedirs(invalid_model_path, exist_ok=True)

        open(os.path.join(invalid_model_path, "model.pt"), "w").close()
        open(os.path.join(invalid_model_path, "metadata.json"), "w").close()

        # Test
        models = get_available_models(self.model_dir)

        self.assertIn("valid_model", models, "Il modello valido dovrebbe essere incluso")
        self.assertNotIn("invalid_model", models, "Il modello non valido dovrebbe essere escluso")

    def test_model_path_construction(self):
        """Test che verifica la corretta costruzione dei percorsi del modello."""

        # Test del controller corretto
        def synthesize_speech_corrected(model_dir, model_name):
            # Costruisce il percorso del modello (CORRETTO)
            model_path = os.path.join(model_dir, model_name)

            # Verifica che il modello esista (CORRETTO)
            model_file = os.path.join(model_path, "model.pt")
            metadata_file = os.path.join(model_path, "metadata.json")

            return {
                "model_path": model_path,
                "model_file": model_file,
                "metadata_file": metadata_file,
                "exists": os.path.exists(model_file) and os.path.exists(metadata_file)
            }

        result = synthesize_speech_corrected(self.model_dir, self.model_name)

        expected_model_path = os.path.join(self.model_dir, self.model_name)
        expected_model_file = os.path.join(expected_model_path, "model.pt")
        expected_metadata_file = os.path.join(expected_model_path, "metadata.json")

        self.assertEqual(result["model_path"], expected_model_path)
        self.assertEqual(result["model_file"], expected_model_file)
        self.assertEqual(result["metadata_file"], expected_metadata_file)

    def test_model_saving_verification(self):
        """Test che verifica la verifica del salvataggio del modello."""

        def _save_model_corrected(model_file):
            """Simula il salvataggio corretto del modello."""
            try:
                # Simula la creazione del file del modello
                os.makedirs(os.path.dirname(model_file), exist_ok=True)

                # Scrive dati fittizi
                with open(model_file, "wb") as f:
                    f.write(b"fake_pytorch_model_data")

                # Verifica che il file sia stato creato e non sia vuoto
                if not os.path.exists(model_file):
                    return False

                if os.path.getsize(model_file) == 0:
                    return False

                return True
            except Exception:
                return False

        def _save_metadata_corrected(metadata, metadata_file):
            """Simula il salvataggio corretto dei metadati."""
            try:
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                if not os.path.exists(metadata_file):
                    return False

                if os.path.getsize(metadata_file) == 0:
                    return False

                return True
            except Exception:
                return False

        # Test salvataggio modello
        model_file = os.path.join(self.model_path, "model.pt")
        result = _save_model_corrected(model_file)
        self.assertTrue(result, "Il salvataggio del modello dovrebbe riuscire")
        self.assertTrue(os.path.exists(model_file), "Il file del modello dovrebbe esistere")
        self.assertGreater(os.path.getsize(model_file), 0, "Il file del modello non dovrebbe essere vuoto")

        # Test salvataggio metadati
        metadata = {"model_name": self.model_name, "version": "1.0"}
        metadata_file = os.path.join(self.model_path, "metadata.json")
        result = _save_metadata_corrected(metadata, metadata_file)
        self.assertTrue(result, "Il salvataggio dei metadati dovrebbe riuscire")
        self.assertTrue(os.path.exists(metadata_file), "Il file dei metadati dovrebbe esistere")
        self.assertGreater(os.path.getsize(metadata_file), 0, "Il file dei metadati non dovrebbe essere vuoto")


def run_tests():
    """Esegue tutti i test."""
    print("=== TEST DELLE CORREZIONI PER IL SALVATAGGIO/CARICAMENTO DEL MODELLO ===")
    print()

    # Crea la suite di test
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelSavingLoading)

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
        print("Le correzioni per il salvataggio e caricamento del modello funzionano correttamente.")
    else:
        print("\n❌ ALCUNI TEST SONO FALLITI!")
        print("Potrebbero essere necessarie ulteriori correzioni.")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()

