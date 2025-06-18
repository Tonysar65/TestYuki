"""
Applicazione console per AI Parlante.
Permette di utilizzare l'applicazione da riga di comando senza interfaccia grafica.
"""

import argparse
import logging
import os
import time


class ConsoleApp:
    """Applicazione console per AI Parlante."""

    def __init__(self, controller):
        """
        Inizializza l'applicazione console.

        Args:
            controller: Controller dell'applicazione
        """
        self.logger = logging.getLogger("YukiAI.console_app")
        self.controller = controller
        self.logger.info("Applicazione console inizializzata")

    def run(self):
        """Esegue l'applicazione console."""
        try:
            print("AI Parlante - Modalità interattiva")
            print("Digita 'help' per visualizzare i comandi disponibili")

            while True:
                command = input("> ").strip()

                if command == "exit" or command == "quit":
                    break
                elif command == "help":
                    self._print_help()
                elif command.startswith("clone"):
                    self._handle_clone_command(command)
                elif command.startswith("synthesize"):
                    self._handle_synthesize_command(command)
                elif command.startswith("recognize"):
                    self._handle_recognize_command(command)
                elif command.startswith("list"):
                    self._handle_list_command(command)
                elif command.startswith("play"):
                    self._handle_play_command(command)
                else:
                    print("Comando non riconosciuto. Digita 'help' per visualizzare i comandi disponibili.")

            return 0
        except Exception as e:
            self.logger.exception(f"Errore durante la modalità interattiva: {e}")
            return 1

    @staticmethod
    def _print_help():
        """Stampa l'aiuto."""
        print("Comandi disponibili:")
        print("  help                                  - Visualizza questo aiuto")
        print("  exit, quit                            - Esci dall'applicazione")
        print("  clone <file_audio> <nome_modello>     - Clona una voce")
        print("  synthesize <nome_modello> <testo>     - Sintetizza il parlato")
        print("  synthesize <nome_modello> -f <file>   - Sintetizza il parlato da un file")
        print("  recognize <file_audio>                - Riconosce il parlato")
        print("  list models                           - Elenca i modelli disponibili")
        print("  play <file_audio>                     - Riproduce un file audio")

    def _handle_clone_command(self, command):
        """Gestisce il comando clone."""
        try:
            parts = command.split()

            if len(parts) < 3:
                print("Parametri insufficienti. Utilizzo: clone <file_audio> <nome_modello>")
                return

            file_audio = parts[1]
            model_name = parts[2]

            if not os.path.exists(file_audio):
                print(f"File non trovato: {file_audio}")
                return

            print(f"Clonazione della voce: {model_name}")
            self.controller.load_reference_audio(file_audio)
            self.controller.train_voice_model(model_name)

            while self.controller.state != self.controller.ProcessState.COMPLETED:
                print(f"Progresso: {int(self.controller.progress * 100)}%")
                time.sleep(1)

            print("Clonazione completata")

        except Exception as e:
            print(f"Errore: {e}")

    def _handle_synthesize_command(self, command):
        """Gestisce il comando synthesize."""
        try:
            parts = command.split()

            if len(parts) < 3:
                print("Parametri insufficienti. Utilizzo: synthesize <nome_modello> <testo>")
                return

            model_name = parts[1]

            if parts[2] == "-f" and len(parts) >= 4:
                file_path = parts[3]

                if not os.path.exists(file_path):
                    print(f"File non trovato: {file_path}")
                    return

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = " ".join(parts[2:])

            print(f"Sintesi vocale: {text[:50]}...")
            self.controller.synthesize_speech(model_name, text)

            while self.controller.state != self.controller.ProcessState.COMPLETED:
                print(f"Progresso: {int(self.controller.progress * 100)}%")
                time.sleep(1)

            print("Sintesi completata")

        except Exception as e:
            print(f"Errore: {e}")

    def _handle_recognize_command(self, command):
        """Gestisce il comando recognize."""
        try:
            parts = command.split()

            if len(parts) < 2:
                print("Parametri insufficienti. Utilizzo: recognize <file_audio>")
                return

            file_audio = parts[1]

            if not os.path.exists(file_audio):
                print(f"File non trovato: {file_audio}")
                return

            from speech_recognition import SpeechRecognizer

            recognizer = SpeechRecognizer()
            print(f"Riconoscimento vocale: {file_audio}")
            text = recognizer.transcribe_file(file_audio)
            print(f"Testo riconosciuto: {text}")

        except Exception as e:
            print(f"Errore: {e}")

    def _handle_list_command(self, command):
        """Gestisce il comando list."""
        try:
            parts = command.split()

            if len(parts) < 2:
                print("Parametri insufficienti. Utilizzo: list <tipo>")
                return

            list_type = parts[1]

            if list_type == "models":
                models = self.controller.get_available_models()

                if models:
                    print("Modelli disponibili:")
                    for model in models:
                        print(f"  - {model}")
                else:
                    print("Nessun modello disponibile")
            else:
                print(f"Tipo non supportato: {list_type}")

        except Exception as e:
            print(f"Errore: {e}")

    def _handle_play_command(self, command):
        """Gestisce il comando play."""
        try:
            parts = command.split()

            if len(parts) < 2:
                print("Parametri insufficienti. Utilizzo: play <file_audio>")
                return

            file_audio = parts[1]

            if not os.path.exists(file_audio):
                print(f"File non trovato: {file_audio}")
                return

            print(f"Riproduzione: {file_audio}")
            self.controller.play_audio(file_audio)

            print("Premi Ctrl+C per interrompere la riproduzione")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.controller.stop_playback()
                print("Riproduzione interrotta")

        except Exception as e:
            print(f"Errore: {e}")


def parse_arguments():
    """Analizza gli argomenti della riga di comando."""
    parser = argparse.ArgumentParser(description="AI Parlante - Applicazione console")

    subparsers = parser.add_subparsers(dest="operation", help="Operazione da eseguire", required=True)

    # Comando clone
    clone_parser = subparsers.add_parser("clone", help="Clona una voce")
    clone_parser.add_argument("input_audio", help="File audio di riferimento")
    clone_parser.add_argument("model_name", help="Nome del modello")
    clone_parser.add_argument("--epochs", type=int, default=100,
                            help="Numero di epoche per l'addestramento (default: 100)")
    clone_parser.add_argument("--batch-size", type=int, default=8,
                            help="Dimensione del batch (default: 8)")
    clone_parser.add_argument("--output-dir", default="models",
                            help="Directory di output per il modello (default: models)")

    # Comando synthesize
    synthesize_parser = subparsers.add_parser("synthesize", help="Sintetizza il parlato")
    synthesize_parser.add_argument("model_name", help="Nome del modello")
    group = synthesize_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("text", nargs="?", help="Testo da sintetizzare")
    group.add_argument("-f", "--file", help="File di testo da sintetizzare")
    synthesize_parser.add_argument("-o", "--output", help="File audio di output")
    synthesize_parser.add_argument("-p", "--play", action="store_true",
                                 help="Riproduci l'audio dopo la sintesi")
    synthesize_parser.add_argument("--speed", type=float, default=1.0,
                                 help="Velocità di riproduzione (default: 1.0)")
    synthesize_parser.add_argument("--pitch", type=float, default=0.0,
                                 help="Modifica del tono in semitoni (default: 0)")

    # Comando recognize
    recognize_parser = subparsers.add_parser("recognize", help="Riconosce il parlato")
    recognize_parser.add_argument("input_audio", help="File audio da trascrivere")
    recognize_parser.add_argument("-o", "--output", help="File di testo di output")
    recognize_parser.add_argument("-m", "--model-type", default="whisper",
                                choices=["whisper", "vosk", "google"],
                                help="Tipo di modello da usare (default: whisper)")
    recognize_parser.add_argument("-l", "--language", default="it",
                                help="Lingua del parlato (default: it)")
    recognize_parser.add_argument("--beam-size", type=int, default=5,
                                help="Dimensione del beam per il riconoscimento (default: 5)")

    # Comando list
    list_parser = subparsers.add_parser("list", help="Elenca risorse disponibili")
    list_parser.add_argument("type", choices=["models", "voices", "languages"],
                           help="Tipo di risorsa da elencare")

    # Comando play
    play_parser = subparsers.add_parser("play", help="Riproduce un file audio")
    play_parser.add_argument("input_audio", help="File audio da riprodurre")
    play_parser.add_argument("-s", "--speed", type=float, default=1.0,
                           help="Velocità di riproduzione (default: 1.0)")
    play_parser.add_argument("-v", "--volume", type=float, default=1.0,
                           help="Volume (0.0-1.0, default: 1.0)")

    # Argomenti globali
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Abilita output verboso")
    parser.add_argument("--log-file", help="File di log")

    return parser.parse_args()