# AI Parlante con Riferimento Vocale

Un'applicazione Python per la creazione di un'intelligenza artificiale parlante che utilizza file audio come riferimento vocale. Questo progetto permette di clonare una voce da un file audio e utilizzarla per sintetizzare nuovo parlato a partire da testo.

## Requisiti di Sistema

- **Sistema Operativo**: Windows 10
- **GPU**: NVIDIA GeForce GTX 1060 3GB o superiore
- **Python**: 3.9
- **CUDA Toolkit**: 11.3 o compatibile
- **cuDNN**: Versione compatibile con CUDA Toolkit

## Installazione

### 1. Installazione di Python 3.9

Se non hai già Python 3.9 installato:

1. Scarica Python 3.9 dal [sito ufficiale](https://www.python.org/downloads/release/python-390/)
2. Durante l'installazione, assicurati di selezionare "Add Python 3.9 to PATH"
3. Completa l'installazione seguendo le istruzioni a schermo

### 2. Installazione di CUDA Toolkit e cuDNN

Per sfruttare l'accelerazione GPU NVIDIA:

1. Scarica e installa [NVIDIA CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)
2. Scarica [cuDNN compatibile con CUDA 11.3](https://developer.nvidia.com/cudnn)
3. Estrai i file cuDNN e copia i contenuti nelle rispettive directory CUDA

### 3. Configurazione dell'Ambiente Virtuale

```bash
# Crea una directory per il progetto (se non stai già usando quella scaricata)
mkdir ai_parlante
cd ai_parlante

# Crea un ambiente virtuale
python -m venv venv

# Attiva l'ambiente virtuale
venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

### 4. Installazione di NVIDIA NeMo (opzionale, per modelli avanzati)

```bash
pip install nemo_toolkit[all]
```

## Struttura del Progetto

```
ai_parlante/
│
├── audio_input/         # Directory per i file audio di riferimento
├── audio_output/        # Directory per i file audio generati
├── data/                # Directory per i dati di addestramento
├── gui/                 # Interfaccia utente grafica
├── models/              # Moduli per i modelli di sintesi e riconoscimento
├── utils/               # Utilità varie
├── voice_models/        # Modelli vocali addestrati
│
├── main.py              # Script principale
├── controller.py        # Controller principale
├── audio_preprocessing.py  # Preprocessing audio
├── feature_extraction.py   # Estrazione caratteristiche
├── voice_model_trainer.py  # Addestramento modello vocale
├── voice_synthesis.py      # Sintesi vocale
├── audio_playback.py       # Riproduzione audio
├── requirements.txt     # Dipendenze Python
└── README.md            # Questo file
```

## Utilizzo

### Avvio dell'Applicazione

```bash
# Assicurati che l'ambiente virtuale sia attivato
venv\Scripts\activate

# Avvia l'applicazione
python main.py
```

### Clonazione di una Voce

1. Avvia l'applicazione
2. Nella scheda "Clonazione Vocale", carica un file audio di riferimento
3. Clicca su "Analizza Audio" per estrarre le caratteristiche vocali
4. Clicca su "Addestra Modello" per creare un modello vocale personalizzato
5. Attendi il completamento dell'addestramento

### Sintesi Vocale

1. Nella scheda "Sintesi Vocale", seleziona il modello vocale addestrato
2. Inserisci il testo da sintetizzare
3. Clicca su "Sintetizza" per generare l'audio
4. Utilizza i controlli di riproduzione per ascoltare l'audio generato
5. Clicca su "Salva" per salvare l'audio generato come file

## Risoluzione dei Problemi

### Errori CUDA

- Assicurati che i driver NVIDIA siano aggiornati
- Verifica che CUDA Toolkit e cuDNN siano installati correttamente
- Controlla che la versione di PyTorch sia compatibile con la tua versione di CUDA

### Problemi di Memoria

La GTX 1060 3GB ha una memoria limitata. Se riscontri errori di memoria:

- Riduci la dimensione del batch durante l'addestramento
- Utilizza modelli più leggeri o versioni quantizzate
- Chiudi altre applicazioni che utilizzano la GPU

### Problemi Audio

- Verifica che i dispositivi audio siano configurati correttamente
- Assicurati che PyAudio sia installato correttamente
- Su Windows, potrebbe essere necessario installare Visual C++ Build Tools

## Licenza

Questo progetto è distribuito con licenza MIT. Vedi il file LICENSE per maggiori dettagli.

## Riconoscimenti

Questo progetto utilizza diverse librerie open source e modelli pre-addestrati. Tutti i riconoscimenti sono elencati nel file CREDITS.md.

