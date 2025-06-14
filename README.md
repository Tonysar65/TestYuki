# AI Parlante con Riferimento Vocale

Un'applicazione Python per la creazione di un'intelligenza artificiale parlante che utilizza file audio come riferimento vocale. Questo progetto permette di clonare una voce da un file audio e utilizzarla per sintetizzare nuovo parlato a partire da testo.

## Requisiti minimi di sistema

- **Sistema Operativo**: Windows 10
- **GPU**: NVIDIA GeForce GTX 1060 3GB o superiore
- **Python**: 3.9 o superiore max 3.11
- **CUDA Toolkit**: 11.3 o compatibile
- **cuDNN**: Versione compatibile con CUDA Toolkit

## Installazione

```bash
pip install -r requirements.txt
```

### 1. Installazione di Python 3.9 o superiore max 3.11

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
mkdir TestYukiAi
cd TestYukiAi

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
TestYuki/
├── audio_input/                  # Moduli per l'acquisizione audio
├── docs/                         # Documentazione del progetto
├── gui/                          # Interfaccia grafica utente
├── models/                       # Modelli di intelligenza artificiale
├── templates/                    # Template per la generazione di modelli
├── voice_models/                 # Modelli vocali
│   └── example_model/            # Esempio di modello vocale
├── LICENSE                       # Licenza del progetto
├── LICENSE.md                    # Dettagli sulla licenza
├── README.md                     # Introduzione e guida del progetto
├── __init__.py                   # Inizializzazione del pacchetto
├── app.py                        # Applicazione principale
├── audio_playback.py             # Riproduzione audio
├── audio_preprocessing.py        # Pre-elaborazione audio
├── console_app.py                # Applicazione da riga di comando
├── controller.py                 # Logica di controllo principale
├── correzioni_testyuki.md        # Note sulle correzioni del progetto
├── create_model_pt.py            # Creazione del modello in PyTorch
├── create_model_pt_simulated.py  # Creazione del modello simulato in PyTorch
├── cuda_utils.py                 # Utilità per l'uso di CUDA
└── feature_extraction.py         # Estrazione delle caratteristiche audio
```

## Funzionalità

- Registrazione audio
- Riproduzione audio
- Estrazione delle caratteristiche audio
- Creazione e simulazione di modelli neurali

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

Questo progetto è rilasciato con licenza MIT per uso esclusivamente non commerciale. 
È consentito l'uso personale, educativo e di ricerca, ma è vietato qualsiasi utilizzo commerciale 
senza autorizzazione esplicita.

Vedi il file LICENSE per i dettagli completi.

## Riconoscimenti

Questo progetto utilizza diverse librerie open source e modelli pre-addestrati. Tutti i riconoscimenti sono elencati nel file CREDITS.md.

## Codice

Alcune parti del codice sono state controllate o realizzate in minima parte con l'IA come ChatGPT e DeepSeek.
Il codice risulta avere ancora delle imperfezioni, ci sono molti avvisi da controllare ed eliminare e qualche funzione che deve essere perfezionata.
C'è ancora del codice duplicato da eliminare.