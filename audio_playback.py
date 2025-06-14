"""
Modulo per la riproduzione audio.
Si occupa di riprodurre l'audio generato.
"""

import os
import logging
import threading
import time
import numpy as np
import pyaudio
import wave
from typing import Optional, Union


class AudioPlayback:
    """Classe per la riproduzione audio."""

    def __init__(self, debug: bool = False):
        """
        Inizializza il modulo di riproduzione audio.

        Args:
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.audio_playback")
        self.debug = debug

        # PyAudio
        self.pyaudio = None
        self.stream = None

        # Thread di riproduzione
        self.playback_thread = None
        self.stop_event = threading.Event()

        # Stato
        self.is_playing = False
        self.current_file = None

        self.logger.info("AudioPlayback inizializzato")

    def play(self, file_path: str) -> bool:
        """
        Riproduce un file audio.

        Args:
            file_path: Percorso del file audio

        Returns:
            bool: True se la riproduzione è avviata con successo, False altrimenti
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File non trovato: {file_path}")
            return False

        # Interrompi eventuali riproduzioni in corso
        self.stop()

        try:
            # Avvia un nuovo thread di riproduzione
            self.stop_event.clear()
            self.current_file = file_path
            self.playback_thread = threading.Thread(target=self._playback_thread, args=(file_path,))
            self.playback_thread.daemon = True
            self.playback_thread.start()

            self.logger.info(f"Riproduzione avviata: {os.path.basename(file_path)}")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante l'avvio della riproduzione: {e}")
            return False

    def play_array(self, waveform: np.ndarray, sample_rate: int = 22050) -> bool:
        """
        Riproduce una forma d'onda audio.

        Args:
            waveform: Forma d'onda audio
            sample_rate: Frequenza di campionamento

        Returns:
            bool: True se la riproduzione è avviata con successo, False altrimenti
        """
        # Interrompi eventuali riproduzioni in corso
        self.stop()

        try:
            # Avvia un nuovo thread di riproduzione
            self.stop_event.clear()
            self.current_file = None
            self.playback_thread = threading.Thread(
                target=self._playback_array_thread,
                args=(waveform, sample_rate)
            )
            self.playback_thread.daemon = True
            self.playback_thread.start()

            self.logger.info("Riproduzione array avviata")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante l'avvio della riproduzione array: {e}")
            return False

    def stop(self) -> bool:
        """
        Interrompe la riproduzione in corso.

        Returns:
            bool: True se l'interruzione è riuscita, False altrimenti
        """
        if not self.is_playing:
            return True

        try:
            # Segnala l'interruzione
            self.stop_event.set()

            # Attendi la terminazione del thread
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)

            # Chiudi lo stream
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            # Chiudi PyAudio
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None

            self.is_playing = False
            self.logger.info("Riproduzione interrotta")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante l'interruzione della riproduzione: {e}")
            return False

    def pause(self) -> bool:
        """
        Mette in pausa la riproduzione in corso.

        Returns:
            bool: True se la pausa è riuscita, False altrimenti
        """
        if not self.is_playing or not self.stream:
            return False

        try:
            self.stream.stop_stream()
            self.logger.info("Riproduzione in pausa")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante la pausa della riproduzione: {e}")
            return False

    def resume(self) -> bool:
        """
        Riprende la riproduzione in pausa.

        Returns:
            bool: True se la ripresa è riuscita, False altrimenti
        """
        if not self.stream:
            return False

        try:
            self.stream.start_stream()
            self.logger.info("Riproduzione ripresa")
            return True

        except Exception as e:
            self.logger.error(f"Errore durante la ripresa della riproduzione: {e}")
            return False

    def is_active(self) -> bool:
        """
        Verifica se la riproduzione è attiva.

        Returns:
            bool: True se la riproduzione è attiva, False altrimenti
        """
        return self.is_playing

    def get_current_file(self) -> Optional[str]:
        """
        Ottiene il file attualmente in riproduzione.

        Returns:
            Optional[str]: Percorso del file in riproduzione, None se nessun file è in riproduzione
        """
        return self.current_file

    def _playback_thread(self, file_path: str):
        """
        Thread di riproduzione per file audio.

        Args:
            file_path: Percorso del file audio
        """
        try:
            # Apri il file audio
            wf = wave.open(file_path, 'rb')

            # Inizializza PyAudio
            self.pyaudio = pyaudio.PyAudio()

            # Apri lo stream
            self.stream = self.pyaudio.open(
                format=self.pyaudio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=self._stream_callback
            )

            # Imposta lo stato
            self.is_playing = True

            # Leggi i dati dal file
            data = wf.readframes(1024)

            # Riproduci l'audio
            while data and not self.stop_event.is_set():
                self.stream.write(data)
                data = wf.readframes(1024)

            # Attendi il completamento
            while self.stream.is_active() and not self.stop_event.is_set():
                time.sleep(0.1)

            # Chiudi lo stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            # Chiudi PyAudio
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None

            # Chiudi il file
            wf.close()

            # Imposta lo stato
            self.is_playing = False
            self.logger.info("Riproduzione completata")

        except Exception as e:
            self.logger.error(f"Errore durante la riproduzione: {e}")
            self.is_playing = False

    def _playback_array_thread(self, waveform: np.ndarray, sample_rate: int):
        """
        Thread di riproduzione per array audio.

        Args:
            waveform: Forma d'onda audio
            sample_rate: Frequenza di campionamento
        """
        try:
            # Normalizza l'audio
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val

            # Converti in int16
            waveform_int16 = (waveform * 32767).astype(np.int16)

            # Inizializza PyAudio
            self.pyaudio = pyaudio.PyAudio()

            # Apri lo stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )

            # Imposta lo stato
            self.is_playing = True

            # Riproduci l'audio
            chunk_size = 1024
            for i in range(0, len(waveform_int16), chunk_size):
                if self.stop_event.is_set():
                    break

                chunk = waveform_int16[i:i + chunk_size].tobytes()
                self.stream.write(chunk)

            # Chiudi lo stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            # Chiudi PyAudio
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None

            # Imposta lo stato
            self.is_playing = False
            self.logger.info("Riproduzione array completata")

        except Exception as e:
            self.logger.error(f"Errore durante la riproduzione array: {e}")
            self.is_playing = False

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """
        Callback per lo stream audio.

        Args:
            in_data: Dati in ingresso
            frame_count: Numero di frame
            time_info: Informazioni temporali
            status: Stato

        Returns:
            tuple: (dati, flag)
        """
        if self.stop_event.is_set():
            return None, pyaudio.paComplete

        return in_data, pyaudio.paContinue

