"""
Modulo per il riconoscimento vocale.
Si occupa di trascrivere il parlato in testo.
"""

import logging
import os
import threading
from typing import Callable

import numpy as np
import torch


class SpeechRecognizer:
    """Classe per il riconoscimento vocale."""

    def __init__(self, model_type: str = "whisper", language: str = "it", debug: bool = False):
        """
        Inizializza il riconoscitore vocale.

        Args:
            model_type: Tipo di modello da utilizzare ("whisper", "vosk", "google")
            language: Lingua del riconoscimento
            debug: Modalità debug
        """
        self.logger = logging.getLogger("YukiAI.speech_recognizer")
        self.model_type = model_type
        self.language = language
        self.debug = debug

        # Verifica se CUDA è disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modello
        self.model = None

        # Thread di riconoscimento
        self.recognition_thread = None
        self.stop_event = threading.Event()

        self.logger.info(
            f"SpeechRecognizer inizializzato (model_type={model_type}, language={language}, device={self.device})")

        # Inizializza il modello
        self._initialize_model()

    def _initialize_model(self):
        """Inizializza il modello di riconoscimento vocale."""
        try:
            if self.model_type == "whisper":
                self._initialize_whisper()
            elif self.model_type == "vosk":
                self._initialize_vosk()
            elif self.model_type == "google":
                self._initialize_google()
            else:
                self.logger.error(f"Tipo di modello non supportato: {self.model_type}")
                raise ValueError(f"Tipo di modello non supportato: {self.model_type}")

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione del modello: {e}")
            raise

    def _initialize_whisper(self):
        """Inizializza il modello Whisper."""
        try:
            import whisper

            self.logger.info("Inizializzazione del modello Whisper")

            # Carica il modello
            model_size = "small"  # Opzioni: tiny, base, small, medium, large
            self.model = whisper.load_model(model_size, device=self.device)

            self.logger.info(f"Modello Whisper {model_size} caricato")

        except ImportError:
            self.logger.error("Libreria whisper non installata")
            raise ImportError("Libreria whisper non installata. Installa con: pip install openai-whisper")

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione del modello Whisper: {e}")
            raise

    def _initialize_vosk(self):
        """Inizializza il modello Vosk."""
        try:
            from vosk import Model, KaldiRecognizer

            self.logger.info("Inizializzazione del modello Vosk")

            # Carica il modello
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "vosk", self.language)

            if not os.path.exists(model_path):
                self.logger.warning(f"Modello Vosk per la lingua {self.language} non trovato. Scaricamento necessario.")

                # In un'implementazione reale, si scaricherebbe il modello
                # Per ora, solleva un'eccezione
                raise FileNotFoundError(f"Modello Vosk per la lingua {self.language} non trovato")

            self.model = Model(model_path)

            self.logger.info(f"Modello Vosk per la lingua {self.language} caricato")

        except ImportError:
            self.logger.error("Libreria vosk non installata")
            raise ImportError("Libreria vosk non installata. Installa con: pip install vosk")

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione del modello Vosk: {e}")
            raise

    def _initialize_google(self):
        """Inizializza il riconoscimento vocale di Google."""
        try:
            import speech_recognition as sr

            self.logger.info("Inizializzazione del riconoscimento vocale di Google")

            # Crea il riconoscitore
            self.model = sr.Recognizer()

            self.logger.info("Riconoscitore Google inizializzato")

        except ImportError:
            self.logger.error("Libreria SpeechRecognition non installata")
            raise ImportError("Libreria SpeechRecognition non installata. Installa con: pip install SpeechRecognition")

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione del riconoscitore Google: {e}")
            raise

    def transcribe_file(self, file_path: str) -> str:
        """
        Trascrive un file audio.

        Args:
            file_path: Percorso del file audio

        Returns:
            str: Testo trascritto
        """
        self.logger.info(f"Trascrizione del file: {file_path}")

        try:
            if self.model_type == "whisper":
                return self._transcribe_whisper(file_path)
            elif self.model_type == "vosk":
                return self._transcribe_vosk(file_path)
            elif self.model_type == "google":
                return self._transcribe_google(file_path)
            else:
                self.logger.error(f"Tipo di modello non supportato: {self.model_type}")
                return ""

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione: {e}")
            return ""

    def _transcribe_whisper(self, file_path: str) -> str:
        """
        Trascrive un file audio utilizzando Whisper.

        Args:
            file_path: Percorso del file audio

        Returns:
            str: Testo trascritto
        """
        try:
            # Carica l'audio
            result = self.model.transcribe(
                file_path,
                language=self.language,
                fp16=torch.cuda.is_available()
            )

            # Estrai il testo
            text = result["text"]

            self.logger.info(f"Trascrizione completata: {len(text)} caratteri")

            return text

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione con Whisper: {e}")
            return ""

    def _transcribe_vosk(self, file_path: str) -> str:
        """
        Trascrive un file audio utilizzando Vosk.

        Args:
            file_path: Percorso del file audio

        Returns:
            str: Testo trascritto
        """
        try:
            from vosk import KaldiRecognizer
            import wave

            # Apri il file audio
            wf = wave.open(file_path, "rb")

            # Crea il riconoscitore
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            # Leggi l'audio e trascrivilo
            results = []

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                if rec.AcceptWaveform(data):
                    results.append(rec.Result())

            # Aggiungi l'ultimo risultato
            results.append(rec.FinalResult())

            # Estrai il testo
            import json
            text = ""

            for res in results:
                jres = json.loads(res)
                if "text" in jres:
                    text += jres["text"] + " "

            self.logger.info(f"Trascrizione completata: {len(text)} caratteri")

            return text.strip()

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione con Vosk: {e}")
            return ""

    def _transcribe_google(self, file_path: str) -> str:
        """
        Trascrive un file audio utilizzando Google Speech Recognition.

        Args:
            file_path: Percorso del file audio

        Returns:
            str: Testo trascritto
        """
        try:
            import speech_recognition as sr

            # Carica l'audio
            with sr.AudioFile(file_path) as source:
                audio_data = self.model.record(source)

            # Trascrivilo
            text = self.model.recognize_google(audio_data, language=self.language)

            self.logger.info(f"Trascrizione completata: {len(text)} caratteri")

            return text

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione con Google: {e}")
            return ""

    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Trascrive un array audio.

        Args:
            audio_data: Array audio
            sample_rate: Frequenza di campionamento

        Returns:
            str: Testo trascritto
        """
        self.logger.info("Trascrizione dell'array audio")

        try:
            if self.model_type == "whisper":
                return self._transcribe_array_whisper(audio_data, sample_rate)
            elif self.model_type == "vosk":
                return self._transcribe_array_vosk(audio_data, sample_rate)
            elif self.model_type == "google":
                return self._transcribe_array_google(audio_data, sample_rate)
            else:
                self.logger.error(f"Tipo di modello non supportato: {self.model_type}")
                return ""

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione dell'array: {e}")
            return ""

    def _transcribe_array_whisper(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Trascrive un array audio utilizzando Whisper.

        Args:
            audio_data: Array audio
            sample_rate: Frequenza di campionamento

        Returns:
            str: Testo trascritto
        """
        try:
            # Converti in float32 se necessario
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalizza
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Trascrivilo
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                fp16=torch.cuda.is_available()
            )

            # Estrai il testo
            text = result["text"]

            self.logger.info(f"Trascrizione array completata: {len(text)} caratteri")

            return text

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione array con Whisper: {e}")
            return ""

    def _transcribe_array_vosk(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Trascrive un array audio utilizzando Vosk.

        Args:
            audio_data: Array audio
            sample_rate: Frequenza di campionamento

        Returns:
            str: Testo trascritto
        """
        try:
            from vosk import KaldiRecognizer

            # Crea il riconoscitore
            rec = KaldiRecognizer(self.model, sample_rate)
            rec.SetWords(True)

            # Converti in int16
            audio_data_int16 = (audio_data * 32767).astype(np.int16)

            # Trascrivilo
            rec.AcceptWaveform(audio_data_int16.tobytes())
            result = rec.FinalResult()

            # Estrai il testo
            import json
            jres = json.loads(result)
            text = jres.get("text", "")

            self.logger.info(f"Trascrizione array completata: {len(text)} caratteri")

            return text

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione array con Vosk: {e}")
            return ""

    def _transcribe_array_google(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Trascrive un array audio utilizzando Google Speech Recognition.

        Args:
            audio_data: Array audio
            sample_rate: Frequenza di campionamento

        Returns:
            str: Testo trascritto
        """
        try:
            import speech_recognition as sr
            import io
            import wave

            # Converti in int16
            audio_data_int16 = (audio_data * 32767).astype(np.int16)

            # Salva in un file WAV in memoria
            byte_io = io.BytesIO()
            with wave.open(byte_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 2 bytes per sample (16 bit)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data_int16.tobytes())

            # Riavvolgi il buffer
            byte_io.seek(0)

            # Carica l'audio
            with sr.AudioFile(byte_io) as source:
                audio_data = self.model.record(source)

            # Trascrivilo
            text = self.model.recognize_google(audio_data, language=self.language)

            self.logger.info(f"Trascrizione array completata: {len(text)} caratteri")

            return text

        except Exception as e:
            self.logger.error(f"Errore durante la trascrizione array con Google: {e}")
            return ""

    def start_listening(self, callback: Callable[[str], None]):
        """
        Avvia l'ascolto in tempo reale.

        Args:
            callback: Funzione di callback che riceve il testo trascritto
        """
        self.logger.info("Avvio dell'ascolto in tempo reale")

        # Interrompi eventuali ascolti in corso
        self.stop_listening()

        try:
            # Avvia un nuovo thread di ascolto
            self.stop_event.clear()
            self.recognition_thread = threading.Thread(
                target=self._listening_thread,
                args=(callback,)
            )
            self.recognition_thread.daemon = True
            self.recognition_thread.start()

            return True

        except Exception as e:
            self.logger.error(f"Errore durante l'avvio dell'ascolto: {e}")
            return False

    def stop_listening(self):
        """Interrompe l'ascolto in tempo reale."""
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.logger.info("Interruzione dell'ascolto in tempo reale")

            # Segnala l'interruzione
            self.stop_event.set()

            # Attendi la terminazione del thread
            self.recognition_thread.join(timeout=1.0)

            return True

        return False

    def _listening_thread(self, callback: Callable[[str], None]):
        """
        Thread di ascolto in tempo reale.

        Args:
            callback: Funzione di callback che riceve il testo trascritto
        """
        self.logger.info("Thread di ascolto avviato")

        try:
            if self.model_type == "whisper":
                self._whisper_listening_thread(callback)
            elif self.model_type == "vosk":
                self._vosk_listening_thread(callback)
            elif self.model_type == "google":
                self._google_listening_thread(callback)
            else:
                self.logger.error(f"Tipo di modello non supportato per l'ascolto in tempo reale: {self.model_type}")

        except Exception as e:
            self.logger.error(f"Errore nel thread di ascolto: {e}")
        finally:
            self.logger.info("Thread di ascolto terminato")

    def _whisper_listening_thread(self, callback: Callable[[str], None]):
        """Thread di ascolto per Whisper."""
        import sounddevice as sd
        import queue

        sample_rate = 16000  # Whisper richiede 16kHz
        channels = 1
        blocksize = 2048
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            """Callback per la cattura audio."""
            if status:
                self.logger.warning(f"Stato audio: {status}")
            audio_queue.put(indata.copy())

        try:
            self.logger.info("Avvio cattura audio per Whisper")

            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=audio_callback,
                blocksize=blocksize,
                dtype='float32'
            ):
                audio_buffer = []
                max_buffer_seconds = 5  # Massimo 5 secondi di buffer

                while not self.stop_event.is_set():
                    # Acquisisci dati audio
                    try:
                        data = audio_queue.get(timeout=0.5)
                        audio_buffer.append(data)

                        # Se il buffer è abbastanza grande, trascrivilo
                        if len(audio_buffer) >= (sample_rate * max_buffer_seconds) / blocksize:
                            audio_data = np.concatenate(audio_buffer)
                            text = self._transcribe_array_whisper(audio_data, sample_rate)

                            if text:
                                callback(text)

                            # Pulisci il buffer
                            audio_buffer = []

                    except queue.Empty:
                        continue

                # Trascrivi eventuali dati rimanenti
                if audio_buffer:
                    audio_data = np.concatenate(audio_buffer)
                    text = self._transcribe_array_whisper(audio_data, sample_rate)
                    if text:
                        callback(text)

        except Exception as e:
            self.logger.error(f"Errore nell'ascolto Whisper: {e}")
            raise

    def _vosk_listening_thread(self, callback: Callable[[str], None]):
        """Thread di ascolto per Vosk."""
        import sounddevice as sd
        from vosk import KaldiRecognizer

        sample_rate = 16000  # Vosk richiede 16kHz
        channels = 1
        blocksize = 4000

        try:
            self.logger.info("Avvio cattura audio per Vosk")

            # Crea il riconoscitore
            rec = KaldiRecognizer(self.model, sample_rate)
            rec.SetWords(True)

            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                blocksize=blocksize,
                dtype='int16'
            ) as stream:
                while not self.stop_event.is_set():
                    data, overflowed = stream.read(blocksize)

                    if overflowed:
                        self.logger.warning("Overflow del buffer audio")

                    if rec.AcceptWaveform(data):
                        result = rec.Result()
                        import json
                        jres = json.loads(result)
                        if "text" in jres:
                            callback(jres["text"])

                # Ottieni l'ultimo risultato
                final_result = rec.FinalResult()
                import json
                jres = json.loads(final_result)
                if "text" in jres:
                    callback(jres["text"])

        except Exception as e:
            self.logger.error(f"Errore nell'ascolto Vosk: {e}")
            raise

    def _google_listening_thread(self, callback: Callable[[str], None]):
        """Thread di ascolto per Google Speech Recognition."""
        import speech_recognition as sr

        try:
            self.logger.info("Avvio cattura audio per Google")

            recognizer = sr.Recognizer()
            microphone = sr.Microphone()

            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)

                while not self.stop_event.is_set():
                    try:
                        self.logger.debug("In ascolto...")
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)

                        text = recognizer.recognize_google(audio, language=self.language)
                        callback(text)

                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        self.logger.debug("Audio non comprensibile")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Errore durante il riconoscimento: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Errore nell'ascolto Google: {e}")
            raise