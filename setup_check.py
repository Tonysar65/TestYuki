"""
Script di verifica della configurazione per AI Parlante
Compatibile con: Ubuntu 20.04+, Windows 10+
Autore: Yuki Assistant
"""

import warnings
import os
import platform
import sys

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["ALSA_CARD"] = "default"

try:
    import pkg_resources
except ImportError:
    print("✗ Il modulo pkg_resources non è installato. Installa 'setuptools'.")
    sys.exit(1)


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

    if version.startswith("3.9") or version.startswith("3.10"):
        print("✓ Versione Python compatibile")
        return True
    else:
        print("✗ Versione Python non compatibile. Si consiglia Python 3.9.x o 3.10.x")
        return False


def check_os():
    """Verifica il sistema operativo."""
    print_header("Verifica Sistema Operativo")

    os_name = platform.system()
    os_version = platform.release()
    full_version = platform.version()

    print(f"Sistema Operativo: {os_name} {full_version}")

    if os_name == "Windows" and "10" in os_version:
        print("✓ Sistema Operativo compatibile: Windows 10")
        return True
    elif os_name == "Linux":
        # Compatibilità estesa per Ubuntu
        try:
            with open("/etc/os-release") as f:
                lines = f.readlines()
            for line in lines:
                if "Ubuntu" in line and any(ver in line for ver in ["20.04", "22.04", "24.04"]):
                    print("✓ Sistema Operativo compatibile: Ubuntu 20.04+")
                    return True
        except Exception:
            pass
        print("⚠ Sistema Linux rilevato ma versione non riconosciuta. Verifica manualmente.")
        return True
    else:
        print("✗ Sistema Operativo non compatibile. Si consiglia Ubuntu 20.04+ o Windows 10")
        return False


def check_package(package_name, min_version=None):
    """Verifica se un pacchetto è installato e la sua versione."""
    try:
        package = pkg_resources.get_distribution(package_name)
        version = package.version
        if min_version and pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
            print(f"✗ {package_name} versione {version} installata, richiede almeno {min_version}")
            return False
        print(f"✓ {package_name} versione {version} installata")
        return True
    except pkg_resources.DistributionNotFound:
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

    return all(check_package(pkg, ver) for pkg, ver in dependencies)


def check_cuda():
    """Verifica se CUDA è disponibile."""
    print_header("Verifica CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            version = torch.version.cuda
            print(f"✓ CUDA disponibile: True")
            print(f"✓ Dispositivi CUDA: {device_count}")
            print(f"✓ Dispositivo principale: {name}")
            print(f"✓ Versione CUDA: {version}")

            # Memoria GPU
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = meminfo.total // (1024 ** 2)
                print(f"✓ Memoria GPU totale: {total_mb} MB")
                pynvml.nvmlShutdown()
            except Exception:
                print("⚠ pynvml non disponibile: memoria GPU non rilevata")
            return True
        else:
            print("✗ CUDA non disponibile")
            return False
    except ImportError:
        print("✗ PyTorch non installato")
        return False


def check_audio_devices():
    """Verifica i dispositivi audio disponibili."""
    print_header("Verifica Dispositivi Audio")
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        input_devices = []
        output_devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                input_devices.append(info["name"])
            if info["maxOutputChannels"] > 0:
                output_devices.append(info["name"])
        pa.terminate()

        print(f"✓ {len(input_devices)} dispositivi di input audio trovati")
        print(f"✓ {len(output_devices)} dispositivi di output audio trovati")
        return True
    except Exception as e:
        print(f"✗ Errore audio: {e}")
        return False


def check_directories():
    """Verifica che le directory necessarie esistano."""
    print_header("Verifica Directory del Progetto")
    required_dirs = [
        "audio_input",
        "audio_output",
        "models",
        "gui",
        "utils",
        "voice_models/example_model",
        "data"
    ]
    all_ok = True
    for d in required_dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                print(f"✓ Directory '{d}' creata")
            except Exception as e:
                print(f"✗ Impossibile creare '{d}': {e}")
                all_ok = False
        else:
            print(f"✓ Directory '{d}' trovata")
    return all_ok


def main():
    """Funzione principale di verifica."""
    print_header("Verifica Configurazione AI Parlante")

    results = [
        ("Versione Python", check_python_version()),
        ("Sistema Operativo", check_os()),
        ("Dipendenze", check_dependencies()),
        ("CUDA", check_cuda()),
        ("Dispositivi Audio", check_audio_devices()),
        ("Directory del Progetto", check_directories()),
    ]

    print_header("Riepilogo")
    all_passed = True
    for name, result in results:
        status = "✓ Superato" if result else "✗ Fallito"
        print(f"{name}: {status}")
        all_passed &= result

    if all_passed:
        print("\n✓ Tutto pronto. L'ambiente è configurato correttamente.")
        return 0
    else:
        print("\n✗ Alcune verifiche sono fallite. Risolvere prima di procedere.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
