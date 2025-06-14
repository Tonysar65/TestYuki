"""
Modulo per l'ottimizzazione CUDA.
Si occupa di configurare e ottimizzare l'utilizzo della GPU NVIDIA.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union


class CUDAOptimizer:
    """Classe per l'ottimizzazione CUDA."""

    def __init__(self, debug: bool = False):
        """
        Inizializza l'ottimizzatore CUDA.

        Args:
            debug: Modalità debug
        """
        self.logger = logging.getLogger("ai_parlante.cuda_optimizer")
        self.debug = debug

        # Verifica se CUDA è disponibile
        self.cuda_available = torch.cuda.is_available()

        # Informazioni sulla GPU
        self.gpu_info = {}

        # Configurazione
        self.mixed_precision = True
        self.memory_efficient = True

        self.logger.info(f"CUDAOptimizer inizializzato (cuda_available={self.cuda_available})")

        # Inizializza CUDA
        if self.cuda_available:
            self._initialize_cuda()

    def _initialize_cuda(self):
        """Inizializza CUDA."""
        try:
            # Ottieni informazioni sulla GPU
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            self.gpu_info = {
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version()
            }

            # Abilita cuDNN benchmark
            torch.backends.cudnn.benchmark = True

            # Abilita TF32 (se disponibile)
            if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

            # Ottieni informazioni sulla memoria
            try:
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(current_device)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

                self.gpu_info["total_memory"] = info.total
                self.gpu_info["free_memory"] = info.free
                self.gpu_info["used_memory"] = info.used

                nvidia_smi.nvmlShutdown()
            except ImportError:
                self.logger.warning("Libreria nvidia-smi non disponibile")
            except Exception as e:
                self.logger.warning(f"Errore durante il recupero delle informazioni sulla memoria: {e}")

            self.logger.info(f"CUDA inizializzato (device={device_name}, cuda={torch.version.cuda})")

            if self.debug:
                self.logger.debug(f"Informazioni GPU: {self.gpu_info}")

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione di CUDA: {e}")

    def is_cuda_available(self) -> bool:
        """
        Verifica se CUDA è disponibile.

        Returns:
            bool: True se CUDA è disponibile, False altrimenti
        """
        return self.cuda_available

    def get_device(self) -> torch.device:
        """
        Ottiene il dispositivo da utilizzare.

        Returns:
            torch.device: Dispositivo da utilizzare
        """
        return torch.device("cuda" if self.cuda_available else "cpu")

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Ottiene informazioni sulla GPU.

        Returns:
            Dict[str, Any]: Informazioni sulla GPU
        """
        return self.gpu_info

    def optimize_memory_usage(self):
        """Ottimizza l'utilizzo della memoria."""
        if not self.cuda_available:
            return

        try:
            # Libera la cache
            torch.cuda.empty_cache()

            # Imposta la modalità di allocazione della memoria
            if hasattr(torch.cuda, "memory_stats"):
                # PyTorch >= 1.6.0
                torch.cuda.memory_stats()

            self.logger.info("Utilizzo della memoria ottimizzato")

        except Exception as e:
            self.logger.error(f"Errore durante l'ottimizzazione della memoria: {e}")

    def enable_mixed_precision(self, enabled: bool = True):
        """
        Abilita o disabilita la precisione mista.

        Args:
            enabled: True per abilitare, False per disabilitare
        """
        self.mixed_precision = enabled

        if not self.cuda_available:
            return

        try:
            # Abilita la precisione mista
            if enabled:
                if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                    self.logger.info("Precisione mista abilitata")
                else:
                    self.logger.warning("Precisione mista non supportata in questa versione di PyTorch")
            else:
                self.logger.info("Precisione mista disabilitata")

        except Exception as e:
            self.logger.error(f"Errore durante la configurazione della precisione mista: {e}")

    def get_optimal_batch_size(self, model_size_mb: float) -> int:
        """
        Calcola la dimensione ottimale del batch in base alla memoria disponibile.

        Args:
            model_size_mb: Dimensione del modello in MB

        Returns:
            int: Dimensione ottimale del batch
        """
        if not self.cuda_available:
            return 1

        try:
            # Ottieni la memoria disponibile
            free_memory_mb = self.gpu_info.get("free_memory", 0) / (1024 * 1024)

            if free_memory_mb == 0:
                # Fallback: utilizza una stima
                device_name = self.gpu_info.get("device_name", "").lower()

                if "1060" in device_name and "3gb" in device_name:
                    free_memory_mb = 2500  # Stima per GTX 1060 3GB
                elif "1060" in device_name and "6gb" in device_name:
                    free_memory_mb = 5500  # Stima per GTX 1060 6GB
                else:
                    free_memory_mb = 2000  # Valore predefinito

            # Calcola la dimensione del batch
            # Utilizza circa il 70% della memoria disponibile
            memory_per_sample = model_size_mb * 2  # Stima: modello + gradienti + buffer
            max_batch_size = int((free_memory_mb * 0.7) / memory_per_sample)

            # Limita la dimensione del batch
            batch_size = max(1, min(max_batch_size, 32))

            self.logger.info(f"Dimensione ottimale del batch: {batch_size}")
            return batch_size

        except Exception as e:
            self.logger.error(f"Errore durante il calcolo della dimensione del batch: {e}")
            return 1

    def autocast_context(self):
        """
        Crea un contesto per la precisione mista.

        Returns:
            Context manager per la precisione mista
        """
        if self.cuda_available and self.mixed_precision and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp,
                                                                                                   "autocast"):
            return torch.cuda.amp.autocast()
        else:
            # Contesto nullo
            import contextlib
            return contextlib.nullcontext()

    def optimize_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Ottimizza un modello per l'inferenza.

        Args:
            model: Modello da ottimizzare

        Returns:
            torch.nn.Module: Modello ottimizzato
        """
        if not self.cuda_available:
            return model

        try:
            # Imposta il modello in modalità di valutazione
            model.eval()

            # Ottimizza il modello
            if hasattr(torch, "jit") and hasattr(torch.jit, "script"):
                try:
                    # Prova a utilizzare TorchScript
                    model = torch.jit.script(model)
                    self.logger.info("Modello ottimizzato con TorchScript")
                except Exception as e:
                    self.logger.warning(f"Impossibile utilizzare TorchScript: {e}")

            return model

        except Exception as e:
            self.logger.error(f"Errore durante l'ottimizzazione del modello: {e}")
            return model

    def cleanup(self):
        """Libera le risorse."""
        if self.cuda_available:
            try:
                # Libera la cache
                torch.cuda.empty_cache()

                self.logger.info("Risorse CUDA liberate")

            except Exception as e:
                self.logger.error(f"Errore durante la pulizia delle risorse CUDA: {e}")


def get_cuda_optimizer(debug: bool = False) -> CUDAOptimizer:
    """
    Ottiene un'istanza dell'ottimizzatore CUDA.

    Args:
        debug: Modalità debug

    Returns:
        CUDAOptimizer: Istanza dell'ottimizzatore CUDA
    """
    return CUDAOptimizer(debug=debug)

