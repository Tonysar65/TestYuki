# CORREZIONI PER IL REPOSITORY TESTYUKI

## File: controller_fixed.py

Questo file contiene le correzioni per il controller.py originale.

### Problemi corretti:
1. Aggiunto il metodo _save_model mancante
2. Corretta la verifica dell'esistenza del modello
3. Standardizzata la gestione dei percorsi
4. Aggiunto il caricamento esplicito del modello prima della sintesi
5. Migliorata la gestione degli errori

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Controller principale per l'applicazione AI Parlante - VERSIONE CORRETTA.
Coordina i vari moduli e gestisce il flusso di lavoro dell'applicazione.
"""

import os
import logging
import threading
import time
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
    
    def __init__(self, input_audio: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 debug: bool = False):
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
                    
                    # Costruisce il percorso del modello (CORRETTO)
                    model_path = os.path.join(self.model_dir, model_name)
                    
                    # Funzione di callback per aggiornare il progresso
                    def progress_callback(progress):
                        # Scala il progresso tra 0.2 e 0.9
                        scaled_progress = 0.2 + (progress * 0.7)
                        self._update_progress(scaled_progress)
                    
                    # Addestra il modello
                    model_file = self.voice_model.train(
                        features=features,
                        model_name=model_name,
                        model_path=model_path,
                        progress_callback=progress_callback
                    )
                    
                    # Verifica che il modello sia stato salvato correttamente (NUOVO)
                    if not self._verify_model_saved(model_path):
                        raise Exception("Il modello non è stato salvato correttamente")
                    
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
        # Costruisce il percorso del modello (CORRETTO)
        model_path = os.path.join(self.model_dir, model_name)
        
        # Verifica che il modello esista (CORRETTO)
        if not self._verify_model_exists(model_path):
            self._update_status(f"Errore: Modello {model_name} non trovato")
            return False
        
        if not text:
            self._update_status("Errore: Nessun testo da sintetizzare")
            return False
        
        self._update_state(ProcessState.SYNTHESIZING)
        self._update_status("Sintesi vocale in corso")
        
        try:
            # Esegui in un thread separato per non bloccare l'UI
            def worker():
                try:
                    # Carica il modello (NUOVO)
                    if not self._load_model_for_synthesis(model_path):
                        raise Exception("Impossibile caricare il modello per la sintesi")
                    
                    # Sintetizza l'audio
                    audio_data = self.synthesizer.synthesize(
                        model_name=model_name,
                        text=text,
                        model_path=model_path
                    )
                    
                    # Salva l'audio se richiesto
                    if output_file:
                        output_path = output_file
                    else:
                        output_path = os.path.join(self.output_dir, f"synthesis_{int(time.time())}.wav")
                    
                    self.audio_processor.save_audio(audio_data, output_path)
                    
                    self._update_state(ProcessState.COMPLETED)
                    self._update_status(f"Sintesi completata: {os.path.basename(output_path)}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Errore durante la sintesi: {e}")
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

    def _verify_model_exists(self, model_path: str) -> bool:
        """
        Verifica che il modello esista e sia valido (NUOVO METODO).
        
        Args:
            model_path: Percorso del modello
            
        Returns:
            bool: True se il modello esiste ed è valido
        """
        try:
            # Verifica che la directory del modello esista
            if not os.path.exists(model_path):
                return False
            
            # Verifica che il file del modello esista
            model_file = os.path.join(model_path, "model.pt")
            if not os.path.exists(model_file):
                return False
            
            # Verifica che il file dei metadati esista
            metadata_file = os.path.join(model_path, "metadata.json")
            if not os.path.exists(metadata_file):
                return False
            
            # Verifica che i file non siano vuoti
            if os.path.getsize(model_file) == 0:
                return False
            
            if os.path.getsize(metadata_file) == 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore durante la verifica del modello: {e}")
            return False

    def _verify_model_saved(self, model_path: str) -> bool:
        """
        Verifica che il modello sia stato salvato correttamente (NUOVO METODO).
        
        Args:
            model_path: Percorso del modello
            
        Returns:
            bool: True se il modello è stato salvato correttamente
        """
        return self._verify_model_exists(model_path)

    def _load_model_for_synthesis(self, model_path: str) -> bool:
        """
        Carica il modello per la sintesi (NUOVO METODO).
        
        Args:
            model_path: Percorso del modello
            
        Returns:
            bool: True se il caricamento è riuscito
        """
        try:
            model_file = os.path.join(model_path, "model.pt")
            return self.voice_model.load_model(model_file)
        except Exception as e:
            self.logger.error(f"Errore durante il caricamento del modello: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """
        Ottiene la lista dei modelli vocali disponibili (MIGLIORATO).
        
        Returns:
            List[str]: Lista dei nomi dei modelli disponibili
        """
        models = []
        try:
            if not os.path.exists(self.model_dir):
                return models
            
            for item in os.listdir(self.model_dir):
                model_path = os.path.join(self.model_dir, item)
                if os.path.isdir(model_path) and self._verify_model_exists(model_path):
                    models.append(item)
            
            return sorted(models)
            
        except Exception as e:
            self.logger.error(f"Errore durante il recupero dei modelli: {e}")
            return []

    def _run_in_thread(self, worker):
        """Esegue una funzione in un thread separato."""
        self.stop_requested = False
        self.worker_thread = threading.Thread(target=worker)
        self.worker_thread.start()

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
            
   
(Content truncated due to size limit. Use line ranges to read in chunks)