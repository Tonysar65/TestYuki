#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Applicazione console per AI Parlante.
Permette di utilizzare l'applicazione da riga di comando senza interfaccia grafica.
"""

import os
import sys
import logging
import argparse
import time
from typing import Dict, Any, List, Tuple, Optional, Union


class ConsoleApp:
    """Applicazione console per AI Parlante."""

    def __init__(self, args):
        """
        Inizializza l'applicazione console.

        Args:
            args: Argomenti della riga di comando
        """
        self.logger = logging.getLogger("ai_parlante.console_app")
        self.args = args

        # Controller
        self.controller = None

        self.logger.info("Applicazione console inizializzata")

    def run(self):
        """
        Esegue l'applicazione console.

        Returns:
            int: Codice di uscita
        """
        try:
            # Ottieni il controller
            from controller import Controller
            self.controller = self.args.controller

            # Verifica l'operazione richiesta
            if hasattr(self.args, "operation"):
                operation = self.args.operation

                if operation == "clone":
                    return self._clone_voice()
                elif operation == "synthesize":
                    return self._synthesize_speech()
                elif operation == "recognize":
                    return self._recognize_speech()
                else:
                    self.logger.error(f"Operazione non supportata: {operation}")
                    return 1
            else:
                # Avvia la modalità interattiva
                return self._interactive_mode()

        except Exception as e:
            self.logger.exception(f"Errore durante l'esecuzione dell'applicazione console: {e}")
            return 1

    def _clone_voice(self):
        """
        Clona una voce a partire da un file audio di riferimento.

        Returns:
            int: Codice di uscita
        """
        try:
            # Verifica i parametri
            if not hasattr(self.args, "input_audio") or not self.args.input_audio:
                self.logger.error("File audio di riferimento non specificato")
                return 1

            if not hasattr(self.args, "model_name") or not self.args.model_name:
                self.logger.error("Nome del modello non specificato")
                return 1

            # Carica il file audio
            print(f"Caricamento del file audio: {self.args.input_audio}")
            self.controller.load_reference_audio(self.args.input_audio)

            # Clona la voce
            print(f"Clonazione della voce: {self.args.model_name}")
            self.controller.train_voice_model(self.args.model_name)

            # Attendi il completamento
            while self.controller.state != self.controller.ProcessState.COMPLETED:
                print(f"Progresso: {int(self.controller.progress * 100)}%")
                time.sleep(1)

            print("Clonazione completata")
            return 0

        except Exception as e:
            self.logger.exception(f"Errore durante la clonazione della voce: {e}")
            return 1

    def _synthesize_speech(self):
        """
        Sintetizza il parlato a partire dal testo.

        Returns:
            int: Codice di uscita
        """
        try:
            # Verifica i parametri
            if not hasattr(self.args, "model_name") or not self.args.model_name:
                self.logger.error("Nome del modello non specificato")
                return 1

            if not hasattr(self.args, "text") or not self.args.text:
                self.logger.error("Testo da sintetizzare non specificato")
                return 1

            # Sintetizza il parlato
            print(f"Sintesi vocale: {self.args.text}")
            output_file = None

            if hasattr(self.args, "output_file") and self.args.output_file:
                output_file = self.args.output_file

            self.controller.synthesize_speech(self.args.model_name, self.args.text, output_file)

            # Attendi il completamento
            while self.controller.state != self.controller.ProcessState.COMPLETED:
                print(f"Progresso: {int(self.controller.progress * 100)}%")
                time.sleep(1)

            print("Sintesi completata")

            # Riproduci l'audio se richiesto
            if hasattr(self.args, "play") and self.args.play:
                print("Riproduzione audio...")
                # La riproduzione è gestita automaticamente dal controller

            return 0

        except Exception as e:
            self.logger.exception(f"Errore durante la sintesi vocale: {e}")
            return 1

    def _recognize_speech(self):
        """
        Riconosce il parlato da un file audio.

        Returns:
            int: Codice di uscita
        """
        try:
            # Verifica i parametri
            if not hasattr(self.args, "input_audio") or not self.args.input_audio:
                self.logger.error("File audio non specificato")
                return 1

            # Importa il modulo di riconoscimento vocale
            from speech_recognition import SpeechRecognizer

            # Crea il riconoscitore
            recognizer = SpeechRecognizer(
                model_type=self.args.model_type if hasattr(self.args, "model_type") else "whisper",
                language=self.args.language if hasattr(self.args, "language") else "it"
            )

            # Riconosci il parlato
            print(f"Riconoscimento vocale: {self.args.input_audio}")
            text = recognizer.transcribe_file(self.args.input_audio)

            # Stampa il risultato
            print(f"Testo riconosciuto: {text}")

            # Salva il risultato se richiesto
            if hasattr(self.args, "output_file") and self.args.output_file:
                with open(self.args.output_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Risultato salvato: {self.args.output_file}")

            return 0

        except Exception as e:
            self.logger.exception(f"Errore durante il riconoscimento vocale: {e}")
            return 1

    def _interactive_mode(self):
        """
        Avvia la modalità interattiva.

        Returns:
            int: Codice di uscita
        """
        try:
            print("AI Parlante - Modalità interattiva")
            print("Digita 'help' per visualizzare i comandi disponibili")

            while True:
                # Leggi il comando
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
        """
        Gestisce il comando clone.

        Args:
            command: Comando
        """
        try:
            # Analizza il comando
            parts = command.split()

            if len(parts) < 3:
                print("Parametri insufficienti. Utilizzo: clone <file_audio> <nome_modello>")
                return

            file_audio = parts[1]
            model_name = parts[2]

            # Verifica che il file esista
            if not os.path.exists(file_audio):
                print(f"File non trovato: {file_audio}")
                return

            # Clona la voce
            print(f"Clonazione della voce: {model_name}")
            self.controller.load_reference_audio(file_audio)
            self.controller.train_voice_model(model_name)

            # Attendi il completamento
            while self.controller.state != self.controller.ProcessState.COMPLETED:
                print(f"Progresso: {int(self.controller.progress * 100)}%")
                time.sleep(1)

            print("Clonazione completata")

        except Exception as e:
            print(f"Errore: {e}")

    def _handle_synthesize_command(self, command):
        """
        Gestisce il comando synthesize.

        Args:
            command: Comando
        """
        try:
            # Analizza il comando
            parts = command.split()

            if len(parts) < 3:
                print("Parametri insufficienti. Utilizzo: synthesize <nome_modello> <testo>")
                return

            model_name = parts[1]

            # Verifica se il testo è specificato direttamente o tramite file
            if parts[2] == "-f" and len(parts) >= 4:
                # Leggi il testo da un file
                file_path = parts[3]

                if not os.path.exists(file_path):
                    print(f"File non trovato: {file_path}")
                    return

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # Utilizza il testo specificato direttamente
                text = " ".join(parts[2:])

            # Sintetizza il parlato
            print(f"Sintesi vocale: {text[:50]}...")
            self.controller.synthesize_speech(model_name, text)

            # Attendi il completamento
            while self.controller.state != self.controller.ProcessState.COMPLETED:
                print(f"Progresso: {int(self.controller.progress * 100)}%")
                time.sleep(1)

            print("Sintesi completata")

        except Exception as e:
            print(f"Errore: {e}")

    @staticmethod
    def _handle_recognize_command(command):
        """
        Gestisce il comando recognize.

        Args:
            command: Comando
        """
        try:
            # Analizza il comando
            parts = command.split()

            if len(parts) < 2:
                print("Parametri insufficienti. Utilizzo: recognize <file_audio>")
                return

            file_audio = parts[1]

            # Verifica che il file esista
            if not os.path.exists(file_audio):
                print(f"File non trovato: {file_audio}")
                return

            # Importa il modulo di riconoscimento vocale
            from speech_recognition import SpeechRecognizer

            # Crea il riconoscitore
            recognizer = SpeechRecognizer()

            # Riconosci il parlato
            print(f"Riconoscimento vocale: {file_audio}")
            text = recognizer.transcribe_file(file_audio)

            # Stampa il risultato
            print(f"Testo riconosciuto: {text}")

        except Exception as e:
            print(f"Errore: {e}")

    def _handle_list_command(self, command):
        """
        Gestisce il comando list.

        Args:
            command: Comando
        """
        try:
            # Analizza il comando
            parts = command.split()

            if len(parts) < 2:
                print("Parametri insufficienti. Utilizzo: list <tipo>")
                return

            list_type = parts[1]

            if list_type == "models":
                # Elenca i modelli disponibili
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
        """
        Gestisce il comando play.

        Args:
            command: Comando
        """
        try:
            # Analizza il comando
            parts = command.split()

            if len(parts) < 2:
                print("Parametri insufficienti. Utilizzo: play <file_audio>")
                return

            file_audio = parts[1]

            # Verifica che il file esista
            if not os.path.exists(file_audio):
                print(f"File non trovato: {file_audio}")
                return

            # Riproduci il file audio
            print(f"Riproduzione: {file_audio}")
            self.controller.play_audio(file_audio)

            # Attendi la fine della riproduzione
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
    """
    Analizza gli argomenti della riga di comando.

    Returns:
        argparse.Namespace: Argomenti analizzati
    """
    parser = argparse.ArgumentParser(description="AI Parlante - Applicazione console")

    # Sottocomandi
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