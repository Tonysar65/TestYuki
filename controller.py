""""
Controller principale per l'applicazione AI Parlante.
Coordina i vari moduli e gestisce il flusso di lavoro dell'applicazione.
"""
import json
import os
import logging
import pickle
import threading
import time
import sounddevice as sd
import soundfile as sf
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

class ProcessState(Enum):
    """Stati possibili per i processi dell'applicazione."""
    IDLE = 0
    LOADING = 1
    PROCESSING = 2
    TRAINING = 3
    SYNTHESIZING = 4
    COMPLETED = 5
    ERROR = 6


class Controller:
    """Controller principale dell'applicazione."""

    def __init__(self, input_audio=None, output_dir="output", model_dir="models", debug=False):
        self.input_audio = input_audio
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.debug = debug
        self.processing_cancelled = False
        self.settings_file = "settings.json"
        self.settings = {}
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

        """
        Inizializza il controller.

        Args:
            input_audio: Percorso del file audio di riferimento
            output_dir: Directory per i file di output
            model_dir: Directory per i modelli vocali
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.controller")
        self.logger.info("Inizializzazione del controller")

        # Configurazione percorsi
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_audio = input_audio
        self.output_dir = output_dir or os.path.join(self.base_dir, "audio_output")
        self.model_dir = model_dir or os.path.join(self.base_dir, "voice_models")
        self.data_dir = os.path.join(self.base_dir, "data")

        # Assicura che le directory esistano
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Stato dell'applicazione
        self.state = ProcessState.IDLE
        self.progress = 0.0
        self.status_message = "Pronto"
        self.debug = debug

        # Callback per aggiornamenti UI
        self.on_state_changed = None
        self.on_progress_changed = None
        self.on_status_changed = None

        # Moduli
        self.audio_processor = None
        self.feature_extractor = None
        self.voice_model = None
        self.synthesizer = None
        self.playback = None

        # Thread di lavoro
        self.worker_thread = None
        self.stop_requested = False

        # Inizializza i moduli
        self._initialize_modules()

        #
        self.processing_cancelled = False  # reset per la prossima esecuzione

    def _initialize_modules(self):
        """Inizializza i vari moduli dell'applicazione."""
        try:
            # Importa i moduli solo quando necessario
            from audio_preprocessing import AudioProcessor
            from feature_extraction import FeatureExtractor
            from voice_model_trainer import VoiceModel
            from voice_synthesis import Synthesizer
            from audio_playback import AudioPlayback

            # Inizializza i moduli
            self.audio_processor = AudioProcessor(debug=self.debug)
            self.feature_extractor = FeatureExtractor(debug=self.debug)
            self.voice_model = VoiceModel(model_dir=self.model_dir, debug=self.debug)
            self.synthesizer = Synthesizer(model_dir=self.model_dir, debug=self.debug)
            self.playback = AudioPlayback(debug=self.debug)

            self.logger.info("Moduli inizializzati con successo")
        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione dei moduli: {e}")
            self._update_state(ProcessState.ERROR)
            self._update_status(f"Errore: {str(e)}")

    def _update_state(self, state: ProcessState):
        """
        Aggiorna lo stato del processo.

        Args:
            state: Nuovo stato
        """
        self.state = state
        self.logger.debug(f"Stato aggiornato: {state.name}")

        if self.on_state_changed:
            self.on_state_changed(state)

    def _update_progress(self, progress: float):
        """
        Aggiorna il progresso del processo.

        Args:
            progress: Valore del progresso (0.0-1.0)
        """
        self.progress = progress

        if self.on_progress_changed:
            self.on_progress_changed(progress)

    def _update_status(self, message: str):
        """
        Aggiorna il messaggio di stato.

        Args:
            message: Nuovo messaggio di stato
        """
        self.status_message = message
        self.logger.info(message)

        if self.on_status_changed:
            self.on_status_changed(message)

    def load_reference_audio(self, file_path: str) -> bool:
        """
        Carica un file audio di riferimento.

        Args:
            file_path: Percorso del file audio

        Returns:
            bool: True se il caricamento è riuscito, False altrimenti
        """
        if not os.path.exists(file_path):
            self._update_status(f"Errore: File {file_path} non trovato")
            return False

        self._update_state(ProcessState.LOADING)
        self._update_status(f"Caricamento del file audio: {os.path.basename(file_path)}")
        self._update_progress(0.0)

        try:
            # Esegui in un thread separato per non bloccare l'UI
            def worker():
                try:
                    # Carica e preprocessa l'audio
                    audio_data = self.audio_processor.load_audio(file_path)
                    self._update_progress(0.5)

                    # Estrai caratteristiche
                    features = self.feature_extractor.extract_features(audio_data)
                    self._update_progress(1.0)

                    # Salva il riferimento
                    self.input_audio = file_path

                    self._update_state(ProcessState.COMPLETED)
                    self._update_status(f"File audio caricato: {os.path.basename(file_path)}")
                    return True
                except Exception as e:
                    self.logger.error(f"Errore durante il caricamento dell'audio: {e}")
                    self._update_state(ProcessState.ERROR)
                    self._update_status(f"Errore: {str(e)}")
                    return False

            self._run_in_thread(worker)
            return True
        except Exception as e:
            self.logger.error(f"Errore durante l'avvio del caricamento audio: {e}")
            self._update_state(ProcessState.ERROR)
            self._update_status(f"Errore: {str(e)}")
            return False

    def get_settings(self):
        """Restituisce le impostazioni correnti."""
        return self.settings

    def save_settings(self, new_settings=None):
        """Salva le impostazioni in un file JSON."""
        if new_settings:
            self.settings = new_settings
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Errore nel salvataggio delle impostazioni: {e}")

    def load_settings(self):
        """Carica le impostazioni da un file JSON."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Errore nel caricamento delle impostazioni: {e}")

    def train_model(self, model_name, epochs, quality):
        """
        Esegue l'addestramento del modello (esempio semplificato).
        Salva un file .pkl come segnaposto per il modello addestrato.
        """
        model_data = {
            "name": model_name,
            "epochs": epochs,
            "quality": quality,
        }

        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
        except Exception as e:
            raise RuntimeError(f"Errore nel salvataggio del modello: {e}")

    def train_voice_model(self, model_name: str) -> bool:
        """
        Addestra un modello vocale basato sul file audio di riferimento.

        Args:
            model_name: Nome del modello da creare

        Returns:
            bool: True se l'addestramento è avviato con successo, False altrimenti
        """
        if not self.input_audio or not os.path.exists(self.input_audio):
            self._update_status("Errore: Nessun file audio di riferimento caricato")
            return False

        self._update_state(ProcessState.TRAINING)
        self._update_status(f"Addestramento del modello vocale: {model_name}")
        self._update_progress(0.0)

        try:
            # Esegui in un thread separato per non bloccare l'UI
            def worker():
                try:
                    # Carica e preprocessa l'audio
                    audio_data = self.audio_processor.load_audio(self.input_audio)
                    self._update_progress(0.1)

                    # Estrai caratteristiche
                    features = self.feature_extractor.extract_features(audio_data)
                    self._update_progress(0.2)

                    # Addestra il modello
                    model_path = os.path.join(self.model_dir, f"{model_name}")

                    # Funzione di callback per aggiornare il progresso
                    def progress_callback(progress):
                        # Scala il progresso tra 0.2 e 0.9
                        scaled_progress = 0.2 + (progress * 0.7)
                        self._update_progress(scaled_progress)

                    self.voice_model.train(
                        features=features,
                        model_name=model_name,
                        model_path=model_path,
                        progress_callback=progress_callback
                    )

                    self._update_progress(1.0)
                    self._update_state(ProcessState.COMPLETED)
                    self._update_status(f"Modello vocale addestrato: {model_name}")
                    return True
                except Exception as e:
                    self.logger.error(f"Errore durante l'addestramento del modello: {e}")
                    self._update_state(ProcessState.ERROR)
                    self._update_status(f"Errore: {str(e)}")
                    return False

            self._run_in_thread(worker)
            return True
        except Exception as e:
            self.logger.error(f"Errore durante l'avvio dell'addestramento: {e}")
            self._update_state(ProcessState.ERROR)
            self._update_status(f"Errore: {str(e)}")
            return False

    def synthesize_speech(self, model_name: str, text: str, output_file: Optional[str] = None) -> bool:
        """
        Sintetizza il parlato a partire dal testo utilizzando il modello vocale specificato.

        Args:
            model_name: Nome del modello da utilizzare
            text: Testo da sintetizzare
            output_file: Percorso del file di output (opzionale)

        Returns:
            bool: True se la sintesi è avviata con successo, False altrimenti
        """
        model_path = os.path.join(self.model_dir, f"{model_name}")
        if not os.path.exists(model_path):
            self._update_status(f"Errore: Modello {model_name} non trovato")
            return False

        if not text:
            self._update_status("Errore: Nessun testo da sintetizzare")
            return False

        self._update_state(ProcessState.SYNTHESIZING)
        self._update_status(f"Sintesi vocale in corso")
        self._update_progress(0.0)

        try:
            # Esegui in un thread separato per non bloccare l'UI
            def worker(output_file=None):
                try:
                    # Genera un nome di file se non specificato
                    if not output_file:
                        timestamp = int(time.time())
                        output_file = os.path.join(self.output_dir, f"synthesis_{timestamp}.wav")

                    # Funzione di callback per aggiornare il progresso
                    def progress_callback(progress):
                        self._update_progress(progress)

                    # Sintetizza il parlato
                    audio_data = self.synthesizer.synthesize(
                        text=text,
                        model_name=model_name,
                        model_path=model_path,
                        progress_callback=progress_callback
                    )

                    # Salva l'audio
                    self.audio_processor.save_audio(audio_data, output_file)

                    self._update_progress(1.0)
                    self._update_state(ProcessState.COMPLETED)
                    self._update_status(f"Sintesi vocale completata: {os.path.basename(output_file)}")

                    # Riproduci l'audio
                    self.playback.play(output_file)

                    return True
                except Exception as e:
                    self.logger.error(f"Errore durante la sintesi vocale: {e}")
                    self._update_state(ProcessState.ERROR)
                    self._update_status(f"Errore: {str(e)}")
                    return False

            self._run_in_thread(worker)
            return True
        except Exception as e:
            self.logger.error(f"Errore durante l'avvio della sintesi: {e}")
            self._update_state(ProcessState.ERROR)
            self._update_status(f"Errore: {str(e)}")
            return False

    def play_audio(self, file_path: str) -> Optional[bool]:
        """
        Riproduce un file audio.

        Args:
            file_path: Percorso del file audio

        Returns:
            bool: True se la riproduzione è avviata con successo, False altrimenti
        """
        if not os.path.exists(file_path):
            self._update_status(f"Errore: File {file_path} non trovato")
            return False

        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
            return None
        except Exception as e:
            print(f"Errore durante la riproduzione: {e}")
            return None

    def stop_playback(self) -> bool:
        """
        Interrompe la riproduzione audio in corso.

        Returns:
            bool: True se l'interruzione è riuscita, False altrimenti
        """
        try:
            self.playback.stop()
            self._update_status("Riproduzione interrotta")
            return True
        except Exception as e:
            self.logger.error(f"Errore durante l'interruzione della riproduzione: {e}")
            self._update_status(f"Errore: {str(e)}")
            return False

    def get_available_models(self):
        models = []

        if not os.path.isdir(self.model_dir):
            return models

        for name in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, name)
            if os.path.isdir(model_path):
                model_file = os.path.join(model_path, "model.pt")
                metadata_file = os.path.join(model_path, "metadata.json")

                if os.path.exists(model_file) and os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        if "model_name" in metadata:
                            models.append(name)
                    except Exception as e:
                        if self.debug:
                            print(f"Errore nella lettura dei metadati per {name}: {e}")
                        continue  # ignora directory con metadati corrotti

        return models

    def _run_in_thread(self, target_function):
        """
        Esegue una funzione in un thread separato.

        Args:
        target_function: Funzione da eseguire
        """
        # Interrompi eventuali thread in esecuzione
        self.stop_requested = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

            # Avvia un nuovo thread
            self.stop_requested = False
            self.worker_thread = threading.Thread(target=target_function)
            self.worker_thread.daemon = True
            self.worker_thread.start()

    def cleanup(self):
        """Esegue la pulizia delle risorse prima della chiusura."""
        self.logger.info("Pulizia delle risorse")

        # Interrompi eventuali thread in esecuzione
        self.stop_requested = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

        # Ferma la riproduzione audio
        if self.playback:
            self.playback.stop()

        # Libera le risorse CUDA
        if self.voice_model:
            self.voice_model.cleanup()

        if self.synthesizer:
            self.synthesizer.cleanup()

        self.logger.info("Pulizia completata")

    def cancel_processing(self):
        """Metodo per interrompere un'elaborazione in corso."""
        self.processing_cancelled = True
        logging.getLogger("ai_parlante.controller").info("Elaborazione annullata dall'utente")

