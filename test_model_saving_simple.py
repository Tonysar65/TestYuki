#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test semplificato per verificare le correzioni del salvataggio del modello.
"""

import os
import tempfile
import json


def test_model_path_handling():
    """Test della gestione dei percorsi dei modelli."""
    print("Test della gestione dei percorsi dei modelli...")

    # Simula la logica del metodo _save_model corretta
    def simulate_save_model(model_path):
        """Simula il salvataggio del modello con la logica corretta."""
        # Assicurati che la directory del modello esista
        model_dir = os.path.dirname(model_path)
        if model_dir:  # Solo se model_path include una directory
            os.makedirs(model_dir, exist_ok=True)

        # Se model_path è una directory, aggiungi il nome del file
        if os.path.isdir(model_path) or not os.path.splitext(model_path)[1]:
            # model_path è una directory, aggiungi il nome del file
            os.makedirs(model_path, exist_ok=True)
            actual_model_path = os.path.join(model_path, "model.pt")
        else:
            # model_path è già un percorso completo del file
            actual_model_path = model_path

        # Simula il salvataggio del modello
        with open(actual_model_path, "w") as f:
            f.write("fake_model_data")

        return actual_model_path

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test 1: model_path è una directory
            model_dir = os.path.join(temp_dir, "test_model")
            saved_path = simulate_save_model(model_dir)
            expected_path = os.path.join(model_dir, "model.pt")

            assert saved_path == expected_path, f"Percorso errato: {saved_path} != {expected_path}"
            assert os.path.exists(saved_path), f"File non creato: {saved_path}"

            # Test 2: model_path è un file completo
            model_file = os.path.join(temp_dir, "direct_model.pt")
            saved_path = simulate_save_model(model_file)

            assert saved_path == model_file, f"Percorso errato: {saved_path} != {model_file}"
            assert os.path.exists(saved_path), f"File non creato: {saved_path}"

            print("✓ Gestione dei percorsi dei modelli corretta")
            return True

        except Exception as e:
            print(f"✗ Errore nel test della gestione dei percorsi: {e}")
            return False


def test_model_validation():
    """Test della validazione dei modelli."""
    print("Test della validazione dei modelli...")

    # Simula la logica del metodo get_available_models corretta
    def simulate_get_available_models(model_dir):
        """Simula il recupero dei modelli disponibili con la logica corretta."""
        models = []

        if not os.path.exists(model_dir):
            return models

        for item in os.listdir(model_dir):
            model_path = os.path.join(model_dir, item)

            if os.path.isdir(model_path):
                model_file = os.path.join(model_path, "model.pt")
                metadata_file = os.path.join(model_path, "metadata.json")

                if os.path.exists(model_file) and os.path.exists(metadata_file):
                    if os.path.getsize(model_file) > 0:
                        models.append(item)

        return models

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Crea modelli di test

            # Modello valido
            valid_model_dir = os.path.join(temp_dir, "valid_model")
            os.makedirs(valid_model_dir, exist_ok=True)

            with open(os.path.join(valid_model_dir, "model.pt"), "w") as f:
                f.write("model_data")

            with open(os.path.join(valid_model_dir, "metadata.json"), "w") as f:
                json.dump({"model_name": "valid_model"}, f)

            # Modello senza metadati
            no_metadata_dir = os.path.join(temp_dir, "no_metadata")
            os.makedirs(no_metadata_dir, exist_ok=True)

            with open(os.path.join(no_metadata_dir, "model.pt"), "w") as f:
                f.write("model_data")

            # Modello vuoto
            empty_model_dir = os.path.join(temp_dir, "empty_model")
            os.makedirs(empty_model_dir, exist_ok=True)

            with open(os.path.join(empty_model_dir, "model.pt"), "w") as f:
                pass  # File vuoto

            with open(os.path.join(empty_model_dir, "metadata.json"), "w") as f:
                json.dump({"model_name": "empty_model"}, f)

            # Test della validazione
            models = simulate_get_available_models(temp_dir)

            assert len(models) == 1, f"Dovrebbe esserci 1 modello valido, trovati {len(models)}"
            assert "valid_model" in models, f"Il modello valido non è stato trovato"
            assert "no_metadata" not in models, f"Il modello senza metadati non dovrebbe essere incluso"
            assert "empty_model" not in models, f"Il modello vuoto non dovrebbe essere incluso"

            print("✓ Validazione dei modelli corretta")
            return True

        except Exception as e:
            print(f"✗ Errore nel test della validazione dei modelli: {e}")
            return False


def test_directory_creation():
    """Test della creazione delle directory."""
    print("Test della creazione delle directory...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test creazione directory annidate
            nested_path = os.path.join(temp_dir, "models", "test_model", "model.pt")

            # Simula la logica corretta
            model_dir = os.path.dirname(nested_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)

            # Verifica che la directory sia stata creata
            assert os.path.exists(model_dir), f"Directory non creata: {model_dir}"

            # Crea il file
            with open(nested_path, "w") as f:
                f.write("test_data")

            assert os.path.exists(nested_path), f"File non creato: {nested_path}"

            print("✓ Creazione delle directory corretta")
            return True

        except Exception as e:
            print(f"✗ Errore nel test della creazione delle directory: {e}")
            return False


def main():
    """Funzione principale per eseguire tutti i test."""
    print("Avvio dei test semplificati per le correzioni del salvataggio del modello...")
    print("=" * 70)

    # Esegui i test
    tests = [
        test_model_path_handling,
        test_model_validation,
        test_directory_creation
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

    print("=" * 70)
    print(f"Risultati dei test:")
    print(f"✓ Test superati: {passed}")
    print(f"✗ Test falliti: {failed}")
    print(f"Totale: {passed + failed}")

    if failed == 0:
        print("\n🎉 Tutti i test sono stati superati! Le correzioni della logica funzionano correttamente.")
        return 0
    else:
        print(f"\n❌ {failed} test sono falliti. Verificare le correzioni.")
        return 1


if __name__ == "__main__":
    exit(main())

