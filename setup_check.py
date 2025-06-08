#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script di verifica della configurazione per AI Parlante
Questo script verifica che tutte le dipendenze necessarie siano installate
e che la GPU NVIDIA sia riconosciuta correttamente.
"""

import os
import sys
import platform
import subprocess
import importlib
from importlib import util
import pkg_resources


def print_header(message):
    """Stampa un'intestazione formattata."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)


def check_python_version():
    """Verifica la versione di Python."""
    print_header("Verifica Versione Python")

    version = platform.python_version()
    print(f"Versione Python: {version}")

    if version.startswith("3.9"):
        print("✓ Versione Python compatibile")
        return True
    else:
        print("✗ Versione Python non compatibile. Si consiglia Python 3.9.x")
        return False


def check_os():
    """Verifica il sistema operativo."""
    print_header("Verifica Sistema Operativo")

    os_name = platform.system()
    os_version = platform.version()
    print(f"Sistema Operativo: {os_name} {os_version}")

    if os_name == "Windows" and "10" in os_version:
        print("✓ Sistema Operativo compatibile")
        return True
    else:
        print("✗ Sistema Operativo non compatibile. Si consiglia Windows 10")
        return False


def check_package(package_name, min_version=None):
    """Verifica se un pacchetto è installato e la sua versione."""
    try:
        package = pkg_resources.get_distribution(package_name)
        version = package.version
        installed = True
    except pkg_resources.DistributionNotFound:
        version = None
        installed = False

    if installed:
        if min_version and pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
            print(
                f"✗ {package_name} versione {version} installata, ma è richiesta la versione {min_version} o superiore")
            return False
        else:
            print(f"✓ {package_name} versione {version} installata")
            return True
    else:
        print(f"✗ {package_name} non installato")
        return False


def check_dependencies():
    """Verifica le dipendenze principali."""
    print_header("Verifica Dipendenze")

    dependencies = [
        ("numpy", "1.20.0"),
        ("scipy", "1.7.0"),
        ("librosa", "0.8.1"),
        ("torch", "1.9.0"),
        ("torchaudio", "0.9.0"),
        ("PyQt5", "5.15.4"),
        ("SpeechRecognition", "3.8.1"),
        ("pydub", "0.25.1"),
        ("PyAudio", "0.2.11")
    ]

    all_installed = True
    for package, min_version in dependencies:
        if not check_package(package, min_version):
            all_installed = False

    return all_installed


def check_cuda():
    """Verifica se CUDA è disponibile."""
    print_header("Verifica CUDA")

    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            cuda_version = torch.version.cuda

            print(f"✓ CUDA disponibile: {cuda_available}")
            print(f"✓ Dispositivi CUDA: {device_count}")
            print(f"✓ Dispositivo principale: {device_name}")
            print(f"✓ Versione CUDA: {cuda_version}")

            # Verifica memoria GPU
            if device_count > 0:
                try:
                    # Prova a importare pynvml per informazioni dettagliate sulla GPU
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory = info.total / (1024 ** 2)  # Converti in MB
                    print(f"✓ Memoria GPU totale: {total_memory:.0f} MB")

                    if "GTX 1060" in device_name and total_memory < 3000:
                        print("⚠ La memoria GPU potrebbe essere insufficiente per alcuni modelli")

                    pynvml.nvmlShutdown()
                except ImportError:
                    print("⚠ pynvml non installato, impossibile ottenere informazioni dettagliate sulla memoria GPU")
                except Exception as e:
                    print(f"⚠ Errore durante il recupero delle informazioni sulla memoria GPU: {e}")

            return True
        else:
            print("✗ CUDA non disponibile")
            print("  Assicurati che i driver NVIDIA siano installati correttamente")
            print("  Verifica che CUDA Toolkit sia installato")
            return False
    except ImportError:
        print("✗ PyTorch non installato o non compilato con supporto CUDA")
        return False
    except Exception as e:
        print(f"✗ Errore durante la verifica di CUDA: {e}")
        return False


def check_audio_devices():
    """Verifica i dispositivi audio disponibili."""
    print_header("Verifica Dispositivi Audio")

    try:
        import pyaudio
        p = pyaudio.PyAudio()

        input_devices = []
        output_devices = []

        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                input_devices.append(dev_info['name'])
            if dev_info['maxOutputChannels'] > 0:
                output_devices.append(dev_info['name'])

        p.terminate()

        if input_devices:
            print("✓ Dispositivi di input audio trovati:")
            for device in input_devices:
                print(f"  - {device}")
        else:
            print("✗ Nessun dispositivo di input audio trovato")

        if output_devices:
            print("✓ Dispositivi di output audio trovati:")
            for device in output_devices:
                print(f"  - {device}")
        else:
            print("✗ Nessun dispositivo di output audio trovato")

        return len(input_devices) > 0 and len(output_devices) > 0
    except ImportError:
        print("✗ PyAudio non installato")
        return False
    except Exception as e:
        print(f"✗ Errore durante la verifica dei dispositivi audio: {e}")
        return False


def check_directories():
    """Verifica che le directory necessarie esistano."""
    print_header("Verifica Directory del Progetto")

    directories = [
        "audio_input",
        "audio_output",
        "models",
        "gui",
        "utils",
        "voice_models",
        "data"
    ]

    all_exist = True
    for directory in directories:
        if os.path.isdir(directory):
            print(f"✓ Directory '{directory}' trovata")
        else:
            print(f"✗ Directory '{directory}' non trovata")
            try:
                os.makedirs(directory)
                print(f"  Directory '{directory}' creata")
            except Exception as e:
                print(f"  Errore durante la creazione della directory '{directory}': {e}")
                all_exist = False

    return all_exist


def main():
    """Funzione principale."""
    print_header("Verifica Configurazione AI Parlante")

    results = []
    results.append(("Versione Python", check_python_version()))
    results.append(("Sistema Operativo", check_os()))
    results.append(("Dipendenze", check_dependencies()))
    results.append(("CUDA", check_cuda()))
    results.append(("Dispositivi Audio", check_audio_devices()))
    results.append(("Directory del Progetto", check_directories()))

    print_header("Riepilogo")

    all_passed = True
    for name, result in results:
        status = "✓ Superato" if result else "✗ Fallito"
        print(f"{name}: {status}")
        all_passed = all_passed and result

    if all_passed:
        print("\n✓ Tutte le verifiche sono state superate. Il sistema è pronto per l'uso.")
    else:
        print("\n✗ Alcune verifiche sono fallite. Risolvi i problemi prima di utilizzare l'applicazione.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

