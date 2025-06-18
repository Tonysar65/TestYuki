"""
Controller principale per l'applicazione AI Parlante.
Coordina i vari moduli e gestisce il flusso di lavoro dell'applicazione.
"""

import json
import logging
import os
import pickle
import threading
import time
from enum import Enum
from typing import Optional

import sounddevice as sd
import soundfile as sf


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

    def __init__(self, input_audio=None, output_dir="output", model_dir="models", audio_device=None, debug=False):
        """
        Inizializza il controller.

        Args:
            input_audio: Percorso del file audio di riferimento
            output_dir: Directory per i file di output
            model_dir: Directory per i modelli vocali
            audio_device: ID dispositivo audio
            debug: Modalità debug
        """
        self.logger = logging.getLogger("YukiAI.controller")
        self.input_audio = input_audio
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.audio_device = audio_device
        self.debug = debug
        self.processing_cancelled = False
        self.settings_file = "settings.json"
        self.settings = {}

        # Configurazione percorsi
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
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
        """Aggiorna lo stato del processo."""
        self.state = state
        self.logger.debug(f"Stato aggiornato: {state.name}")

        if self.on_state_changed:
            self.on_state_changed(state)

    def _update_progress(self, progress: float):
        """Aggiorna il progresso del processo."""
        self.progress = progress

        if self.on_progress_changed:
            self.on_progress_changed(progress)

    def _update_status(self, message: str):
        """Aggiorna il messaggio di stato."""
        self.status_message = message
        self.logger.info(message)

        if self.on_status_changed:
            self.on_status_changed(message)

    def load_reference_audio(self, file_path: str) -> bool:
        """Carica un file audio di riferimento."""
        if not os.path.exists(file_path):
            self._update_status(f"Errore: File {file_path} non trovato")
            return False

        self._update_state(ProcessState.LOADING)
        self._update_status(f"Caricamento del file audio: {os.path.basename(file_path)}")
        self._update_progress(0.0)

        try:
            def worker():
                try:
                    audio_data = self.audio_processor.load_audio(file_path)
                    self._update_progress(0.5)

                    features = self.feature_extractor.extract_features(audio_data)
                    self._update_progress(1.0)

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
        """Esegue l'addestramento del modello."""
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
        """Addestra un modello vocale basato sul file audio di riferimento."""
        if not self.input_audio or not os.path.exists(self.input_audio):
            self._update_status("Errore: Nessun file audio di riferimento caricato")
            return False

        self._update_state(ProcessState.TRAINING)
        self._update_status(f"Addestramento del modello vocale: {model_name}")
        self._update_progress(0.0)

        try:
            def worker():
                try:
                    audio_data = self.audio_processor.load_audio(self.input_audio)
                    self._update_progress(0.1)

                    features = self.feature_extractor.extract_features(audio_data)
                    self._update_progress(0.2)

                    model_path = os.path.join(self.model_dir, f"{model_name}")

                    def progress_callback(progress):
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
        """Sintetizza il parlato a partire dal testo, usando il modello specificato."""

        # Percorso del modello vocale
        model_path = os.path.join(self.model_dir, model_name)

        # Verifica che il modello esista
        if not os.path.exists(model_path):
            self._update_status(f"Errore: Modello {model_name} non trovato")
            return False

        # Verifica che il testo non sia vuoto
        if not text:
            self._update_status("Errore: Nessun testo da sintetizzare")
            return False

        # Aggiorna lo stato iniziale
        self._update_state(ProcessState.SYNTHESIZING)
        self._update_status("Sintesi vocale in corso")
        self._update_progress(0.0)

        try:
            # Funzione di sintesi eseguita in un thread separato
            def worker():
                try:
                    # Se non specificato, genera il nome automatico del file di output
                    out_file = output_file or os.path.join(
                        self.output_dir, f"synthesis_{int(time.time())}.wav"
                    )

                    # Callback per aggiornare la barra di progresso
                    def progress_callback(progress):
                        self._update_progress(progress)

                    # Avvia la sintesi vocale con il modello selezionato
                    audio_data = self.synthesizer.synthesize(
                        text=text,
                        model_name=model_name,
                        model_path=model_path,
                        progress_callback=progress_callback
                    )

                    # Salva il file audio .wav
                    self.audio_processor.save_audio(audio_data, out_file)

                    # Aggiorna stato e barra di progresso al termine
                    self._update_progress(1.0)
                    self._update_state(ProcessState.COMPLETED)
                    self._update_status(f"Sintesi vocale completata: {os.path.basename(out_file)}")

                    # Riproduce l’audio generato
                    self.playback.play(out_file)
                    return True

                except Exception as e:
                    # Gestione errori durante la sintesi vocale
                    self.logger.error(f"Errore durante la sintesi vocale: {e}")
                    self._update_state(ProcessState.ERROR)
                    self._update_status(f"Errore: {str(e)}")
                    return False

            # Esegue il worker in un thread separato per non bloccare la GUI
            self._run_in_thread(worker)
            return True

        except Exception as e:
            # Gestione errori generali
            self.logger.error(f"Errore durante l'avvio della sintesi: {e}")
            self._update_state(ProcessState.ERROR)
            self._update_status(f"Errore: {str(e)}")
            return False

    def play_audio(self, file_path: str) -> Optional[bool]:
        """Riproduce un file audio."""
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
        """Interrompe la riproduzione audio in corso."""
        try:
            self.playback.stop()
            self._update_status("Riproduzione interrotta")
            return True
        except Exception as e:
            self.logger.error(f"Errore durante l'interruzione della riproduzione: {e}")
            self._update_status(f"Errore: {str(e)}")
            return False

    def get_available_models(self):
        """Restituisce i modelli disponibili."""
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
                        continue

        return models

    def _run_in_thread(self, target_function):
        """Esegue una funzione in un thread separato."""
        self.stop_requested = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

        self.stop_requested = False
        self.worker_thread = threading.Thread(target=target_function)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def cleanup(self):
        """Esegue la pulizia delle risorse prima della chiusura."""
        self.logger.info("Pulizia delle risorse")

        self.stop_requested = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

        if self.playback:
            self.playback.stop()

        if self.voice_model:
            self.voice_model.cleanup()

        if self.synthesizer:
            self.synthesizer.cleanup()

        self.logger.info("Pulizia completata")

    def cancel_processing(self):
        """Interrompe un'elaborazione in corso."""
        self.processing_cancelled = True
        self.logger.info("Elaborazione annullata dall'utente")

    def set_audio_device(self, device_id):
        """Imposta il dispositivo audio di output."""
        try:
            sd.default.device = device_id
            self.audio_device = device_id
            self.logger.info(f"Dispositivo audio impostato su ID {device_id}: {sd.query_devices(device_id)['name']}")
        except Exception as e:
            self.logger.error(f"Errore nell'impostazione del dispositivo audio: {e}")
            raise