# AI Parlante con Riferimento Vocale

Un'applicazione Python per la creazione di un'intelligenza artificiale parlante che utilizza file audio come riferimento vocale. Questo progetto permette di clonare una voce da un file audio e utilizzarla per sintetizzare nuovo parlato a partire da testo.

## Requisiti minimi di sistema

### Hardware
- **GPU**: NVIDIA GeForce GTX 1060 3GB o superiore (consigliato RTX 2060+ per prestazioni migliori)
- **RAM**: 8GB (16GB consigliati)
- **Spazio disco**: 10GB liberi (SSD consigliato)

### Software
- **Python**: 3.9 o superiore (max 3.11)
- **CUDA Toolkit**: 11.3 o compatibile
- **cuDNN**: Versione compatibile con CUDA Toolkit

## Installazione

### Windows 10

#### 1. Installazione di Python
1. Scarica Python 3.9 dal [sito ufficiale](https://www.python.org/downloads/release/python-390/)
2. Durante l'installazione:
   - Seleziona "Add Python 3.9 to PATH"
   - Clicca su "Customize installation" e assicurati che "pip" sia selezionato
3. Completa l'installazione seguendo le istruzioni

#### 2. Installazione driver e toolkit NVIDIA
1. Aggiorna i driver GPU dal [sito NVIDIA](https://www.nvidia.com/Download/index.aspx)
2. Installa [CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-11-3-0-download-archive)
3. Installa [cuDNN compatibile](https://developer.nvidia.com/cudnn) (copiare i file nella cartella CUDA)

#### 3. Configurazione ambiente
```powershell
# Crea e attiva ambiente virtuale
python -m venv venv
venv\Scripts\activate
```

# Installa dipendenze
```
pip install --upgrade pip
pip install -r requirements.txt
```
# Installazione opzionale per modelli avanzati
```
pip install nemo_toolkit[all]
```
Ubuntu 22.04 LTS
1. Preparazione sistema
bash

# Aggiorna sistema
```
sudo apt update && sudo apt upgrade -y
```
# Installa driver NVIDIA
```
sudo ubuntu-drivers autoinstall
sudo reboot
```
# Verifica installazione driver
```
nvidia-smi
```
2. Installazione CUDA e cuDNN
bash

# Installa CUDA 11.3
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-3
```

# Configura environment
```bash
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

# Verifica CUDA
```bash
nvcc --version
```
3. Installazione dipendenze
# Dipendenze di sistema
```bash
sudo apt install -y python3-pip python3-venv build-essential \portaudio19-dev libasound2-dev libjack-dev ffmpeg
```
# Configurazione ambiente Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Utilizzo dell'Applicazione
Avvio
# Windows
```bash
venv\Scripts\activate
python main.py
```

# Linux
```bash
source venv/bin/activate
python main.py
```

Funzionalità principali

    Clonazione Vocale:

        Carica un file audio campione (formati supportati: WAV, MP3, FLAC)

        Analizza le caratteristiche vocali

        Addestra un modello personalizzato

    Sintesi Vocale:

        Seleziona un modello addestrato

        Inserisci il testo da convertire in parlato

        Regola parametri (tono, velocità, enfasi)

        Esporta l'audio generato

    Strumenti Audio:

        Registrazione diretta

        Editing base degli audio

        Analisi spettrale

 Risoluzione dei Problemi comuni Windows

    Errori CUDA: Verificare la compatibilità driver/CUDA con nvidia-smi

    Mancanza MSVC: Installare Build Tools per Visual Studio

    Problemi audio: Reinstallare PyAudio con pip install --force-reinstall pyaudio

Problemi comuni Linux

    Permessi audio:
```bash
sudo usermod -a -G audio $USER
sudo reboot
```

Errori libreria:
```bash
    sudo apt install libsm6 libxext6 libxrender-dev
```

Ottimizzazione prestazioni

    Ridurre la dimensione del batch in config.yaml per GPU con poca VRAM

    Usare modelli quantizzati per sistemi meno potenti

    Chiudere altre applicazioni che utilizzano la GPU durante l'addestramento

Licenza

Questo progetto è rilasciato con licenza MIT per uso esclusivamente non commerciale.
È consentito l'uso personale, educativo e di ricerca, ma è vietato qualsiasi utilizzo commerciale
senza autorizzazione esplicita.

Vedi il file LICENSE per i dettagli completi.
Riconoscimenti

    NVIDIA per CUDA e cuDNN

    PyTorch per il framework di deep learning

    Comunità open source per le librerie audio

    Tutti i contributori elencati in CREDITS.md

Note sullo Sviluppo

Il codice include alcune parti sviluppate con l'assistenza di strumenti AI (ChatGPT, DeepSeek) e presenta:

    Alcuni warning da risolvere

    Funzioni da ottimizzare

    Codice duplicato da eliminare

    Test da implementare

Contributi e pull request sono benvenuti per migliorare il progetto.