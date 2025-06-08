# Riferimento Tecnico - AI Parlante

## Introduzione

Questo documento fornisce una descrizione tecnica dettagliata dell'applicazione AI Parlante, un sistema di clonazione e sintesi vocale basato su tecnologie di intelligenza artificiale. Il documento è destinato agli sviluppatori che desiderano comprendere l'architettura e l'implementazione dell'applicazione, modificarla o estenderla.

## Architettura del Sistema

AI Parlante è strutturato secondo un'architettura modulare che separa le diverse funzionalità in componenti indipendenti. Questo approccio facilita la manutenzione, il testing e l'estensione del sistema.

### Diagramma dell'Architettura

```
+---------------------+     +----------------------+     +----------------------+
|                     |     |                      |     |                      |
|  File Audio di      |---->|  Modulo di           |---->|  Modulo di           |
|  Riferimento        |     |  Preprocessing Audio  |     |  Estrazione          |
|                     |     |                      |     |  Caratteristiche      |
+---------------------+     +----------------------+     +----------------------+
                                                                |
                                                                v
+---------------------+     +----------------------+     +----------------------+
|                     |     |                      |     |                      |
|  Interfaccia        |<--->|  Controller          |<--->|  Modulo di           |
|  Utente Grafica     |     |  Principale          |     |  Addestramento       |
|                     |     |                      |     |  Modello Vocale       |
+---------------------+     +----------------------+     +----------------------+
      ^   |                        ^   |                        ^
      |   |                        |   |                        |
      |   v                        |   v                        v
+---------------------+     +----------------------+     +----------------------+
|                     |     |                      |     |                      |
|  Modulo di          |<--->|  Modulo di           |<--->|  Modello di          |
|  Riproduzione       |     |  Sintesi Vocale      |     |  Voce Addestrato     |
|  Audio              |     |  (TTS con CUDA)      |     |  (su GPU NVIDIA)     |
|                     |     |                      |     |                      |
+---------------------+     +----------------------+     +----------------------+
      ^                              ^
      |                              |
      v                              v
+---------------------+     +----------------------+
|                     |     |                      |
|  Output Audio       |     |  Input Testuale      |
|  (Voce Clonata)     |     |  (Da Sintetizzare)   |
|                     |     |                      |
+---------------------+     +----------------------+
```

### Componenti Principali

#### 1. Controller Principale (`controller.py`)

Il Controller Principale è il componente centrale che coordina tutti gli altri moduli. Implementa il pattern Facade, fornendo un'interfaccia semplificata per le funzionalità complesse del sistema.

Responsabilità:
- Inizializzazione e gestione dei moduli
- Coordinamento del flusso di lavoro
- Gestione dello stato dell'applicazione
- Comunicazione con l'interfaccia utente

#### 2. Modulo di Preprocessing Audio (`audio_preprocessing.py`)

Questo modulo si occupa di caricare, normalizzare e preparare i file audio per l'elaborazione.

Responsabilità:
- Caricamento di file audio in vari formati
- Normalizzazione dell'audio
- Rimozione del silenzio
- Ricampionamento
- Divisione in segmenti

#### 3. Modulo di Estrazione Caratteristiche (`feature_extraction.py`)

Questo modulo estrae le caratteristiche vocali dai file audio per l'addestramento del modello.

Responsabilità:
- Estrazione di mel-spettrogrammi
- Estrazione di MFCC (Mel-Frequency Cepstral Coefficients)
- Estrazione di pitch (F0)
- Estrazione di energia
- Estrazione di caratteristiche prosodiche

#### 4. Modulo di Addestramento Modello Vocale (`voice_model_trainer.py`)

Questo modulo si occupa di addestrare un modello di sintesi vocale basato sulle caratteristiche estratte.

Responsabilità:
- Creazione del modello
- Addestramento del modello
- Salvataggio del modello
- Caricamento del modello

#### 5. Modulo di Sintesi Vocale (`voice_synthesis.py`)

Questo modulo converte il testo in parlato utilizzando il modello vocale addestrato.

Responsabilità:
- Preprocessamento del testo
- Generazione del mel-spettrogramma
- Conversione del mel-spettrogramma in forma d'onda
- Postprocessamento dell'audio

#### 6. Modulo di Clonazione Vocale (`voice_cloning.py`)

Questo modulo si occupa di clonare una voce a partire da un file audio di riferimento.

Responsabilità:
- Estrazione dell'embedding del parlante
- Addestramento del modello di clonazione
- Sintesi vocale con la voce clonata

#### 7. Modulo di Riconoscimento Vocale (`speech_recognition.py`)

Questo modulo si occupa di trascrivere il parlato in testo.

Responsabilità:
- Trascrizione di file audio
- Trascrizione in tempo reale
- Supporto per diversi motori di riconoscimento vocale

#### 8. Modulo di Riproduzione Audio (`audio_playback.py`)

Questo modulo si occupa di riprodurre l'audio generato.

Responsabilità:
- Riproduzione di file audio
- Riproduzione di array audio
- Controllo della riproduzione (pausa, stop)

#### 9. Modulo di Ottimizzazione CUDA (`cuda_utils.py`)

Questo modulo si occupa di configurare e ottimizzare l'utilizzo della GPU NVIDIA.

Responsabilità:
- Verifica della disponibilità di CUDA
- Ottimizzazione dell'utilizzo della memoria
- Configurazione della precisione mista
- Calcolo della dimensione ottimale del batch

#### 10. Interfaccia Utente Grafica (`gui/`)

L'interfaccia utente grafica è implementata utilizzando PyQt5 e fornisce un'interfaccia user-friendly per l'applicazione.

Responsabilità:
- Visualizzazione dell'interfaccia
- Gestione degli eventi utente
- Comunicazione con il controller
- Visualizzazione dei risultati

#### 11. Applicazione Console (`console_app.py`)

L'applicazione console permette di utilizzare AI Parlante da riga di comando, senza interfaccia grafica.

Responsabilità:
- Analisi degli argomenti della riga di comando
- Esecuzione dei comandi
- Visualizzazione dei risultati

## Flusso di Lavoro

### Clonazione Vocale

1. L'utente carica un file audio di riferimento
2. Il file viene preprocessato ed analizzato
3. Le caratteristiche vocali vengono estratte
4. Il modello di sintesi vocale viene addestrato/adattato
5. Il modello viene salvato per uso futuro

### Sintesi Vocale

1. L'utente inserisce il testo da sintetizzare
2. Il controller carica il modello vocale addestrato
3. Il testo viene convertito in parlato usando il modello
4. L'audio generato viene riprodotto e/o salvato

## Implementazione Tecnica

### Modelli di Sintesi Vocale

AI Parlante utilizza diversi modelli di sintesi vocale, implementati utilizzando PyTorch:

#### VoiceEncoder

```python
class VoiceEncoder(nn.Module):
    """
    Encoder per la voce del parlante.
    Converte le caratteristiche audio in un embedding del parlante.
    """
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, 
                embedding_dim: int = 512, num_layers: int = 3):
        """
        Inizializza l'encoder.
        
        Args:
            input_dim: Dimensione dell'input (numero di bande mel)
            hidden_dim: Dimensione dello stato nascosto
            embedding_dim: Dimensione dell'embedding
            num_layers: Numero di layer LSTM
        """
        super(VoiceEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input di forma [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Embedding di forma [batch_size, embedding_dim]
        """
        # LSTM
        output, (hidden, _) = self.lstm(x)
        
        # Prendi l'ultimo stato nascosto di entrambe le direzioni
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Proietta nell'embedding
        embedding = self.linear(hidden)
        embedding = self.relu(embedding)
        
        # Normalizza l'embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
```

#### TextEncoder

```python
class TextEncoder(nn.Module):
    """
    Encoder per il testo.
    Converte il testo in una rappresentazione latente.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512, 
                hidden_dim: int = 512, num_layers: int = 2):
        """
        Inizializza l'encoder.
        
        Args:
            vocab_size: Dimensione del vocabolario
            embedding_dim: Dimensione dell'embedding
            hidden_dim: Dimensione dello stato nascosto
            num_layers: Numero di layer LSTM
        """
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input di forma [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Rappresentazione di forma [batch_size, seq_len, hidden_dim]
        """
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        output, _ = self.lstm(x)
        
        # Proietta
        output = self.linear(output)
        output = self.relu(output)
        
        return output
```

#### Decoder

```python
class Decoder(nn.Module):
    """
    Decoder per la sintesi vocale.
    Converte la rappresentazione latente in mel-spettrogramma.
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, 
                output_dim: int = 80, num_layers: int = 3):
        """
        Inizializza il decoder.
        
        Args:
            input_dim: Dimensione dell'input
            hidden_dim: Dimensione dello stato nascosto
            output_dim: Dimensione dell'output (numero di bande mel)
            num_layers: Numero di layer LSTM
        """
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input di forma [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Output di forma [batch_size, seq_len, output_dim]
        """
        # LSTM
        output, _ = self.lstm(x)
        
        # Proietta nell'output
        output = self.linear(output)
        
        return output
```

#### VoiceCloningModel

```python
class VoiceCloningModel(nn.Module):
    """Modello per la clonazione vocale."""
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 256, output_dim: int = 80):
        """
        Inizializza il modello.
        
        Args:
            input_dim: Dimensione dell'input (numero di coefficienti MFCC)
            hidden_dim: Dimensione dello stato nascosto
            output_dim: Dimensione dell'output (numero di bande mel)
        """
        super(VoiceCloningModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input di forma [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Output di forma [batch_size, seq_len, output_dim]
        """
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        output = self.decoder(encoded)
        
        return output
```

### Ottimizzazione CUDA

AI Parlante utilizza CUDA per accelerare l'addestramento e l'inferenza dei modelli di sintesi vocale. Il modulo `cuda_utils.py` fornisce funzioni per ottimizzare l'utilizzo della GPU NVIDIA:

```python
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
```

```python
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
```

```python
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
      
(Content truncated due to size limit. Use line ranges to read in chunks)