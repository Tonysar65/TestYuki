# Istruzioni di Installazione Dettagliate - AI Parlante

Questo documento fornisce istruzioni dettagliate per l'installazione di AI Parlante su Windows 10 con supporto per NVIDIA GeForce GTX 1060 3GB.

## 1. Installazione di Python 3.9

1. Scarica Python 3.9 dal [sito ufficiale](https://www.python.org/downloads/release/python-390/)
   - Seleziona "Windows installer (64-bit)" per scaricare il file eseguibile

2. Esegui il file scaricato (es. `python-3.9.0-amd64.exe`)

3. Nella schermata di installazione, assicurati di:
   - Selezionare "Add Python 3.9 to PATH"
   - Cliccare su "Customize installation"

4. Nella schermata "Optional Features":
   - Seleziona tutte le opzioni
   - Clicca su "Next"

5. Nella schermata "Advanced Options":
   - Seleziona "Install for all users"
   - Seleziona "Add Python to environment variables"
   - Seleziona "Precompile standard library"
   - Clicca su "Install"

6. Al termine dell'installazione, verifica che Python sia stato installato correttamente:
   - Apri un prompt dei comandi (cmd)
   - Esegui il comando: `python --version`
   - Dovresti vedere: `Python 3.9.0` o simile

## 2. Installazione di NVIDIA CUDA Toolkit 11.3

1. Verifica che i driver NVIDIA siano aggiornati:
   - Apri il Pannello di controllo NVIDIA
   - Verifica la versione del driver
   - Se necessario, scarica e installa i driver più recenti dal [sito NVIDIA](https://www.nvidia.com/Download/index.aspx)

2. Scarica CUDA Toolkit 11.3 dal [sito NVIDIA](https://developer.nvidia.com/cuda-11.3.0-download-archive)
   - Seleziona: Windows > x86_64 > 10 > exe (local)
   - Scarica il file eseguibile

3. Esegui il file scaricato (es. `cuda_11.3.0_465.89_win10.exe`)

4. Nella schermata di installazione:
   - Seleziona "Express (Recommended)"
   - Clicca su "Next"

5. Attendi il completamento dell'installazione

6. Al termine dell'installazione, verifica che CUDA sia stato installato correttamente:
   - Apri un prompt dei comandi (cmd)
   - Esegui il comando: `nvcc --version`
   - Dovresti vedere: `Cuda compilation tools, release 11.3` o simile

## 3. Installazione di cuDNN

1. Registrati o accedi al [NVIDIA Developer Program](https://developer.nvidia.com/cudnn)

2. Scarica cuDNN compatibile con CUDA 11.3
   - Seleziona "Download cuDNN v8.2.1 (June 7th, 2021), for CUDA 11.x"
   - Scarica il file ZIP per Windows 10

3. Estrai il file ZIP scaricato

4. Copia i file nelle rispettive directory CUDA:
   - Copia i file dalla cartella `bin` in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin`
   - Copia i file dalla cartella `include` in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include`
   - Copia i file dalla cartella `lib` in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64`

5. Aggiungi la directory bin di CUDA al PATH di sistema:
   - Apri le Impostazioni di sistema
   - Vai a "Informazioni sul sistema" > "Impostazioni di sistema avanzate" > "Variabili d'ambiente"
   - Nella sezione "Variabili di sistema", seleziona "Path" e clicca su "Modifica"
   - Aggiungi `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin` se non è già presente
   - Clicca su "OK" per chiudere tutte le finestre

## 4. Installazione di AI Parlante

### 4.1. Preparazione dell'Ambiente

1. Crea una directory per l'applicazione:
   ```
   mkdir C:\TestYuki
   ```

2. Estrai il file ZIP di AI Parlante nella directory creata

3. Apri un prompt dei comandi (cmd) e naviga alla directory dell'applicazione:
   ```
   cd C:\TestYuki
   ```

### 4.2. Creazione dell'Ambiente Virtuale

1. Crea un ambiente virtuale Python:
   ```
   python -m venv venv
   ```

2. Attiva l'ambiente virtuale:
   ```
   venv\Scripts\activate
   ```

3. Verifica che l'ambiente virtuale sia attivato:
   - Il prompt dei comandi dovrebbe mostrare `(venv)` all'inizio della riga

### 4.3. Installazione delle Dipendenze

1. Installa le dipendenze richieste:
   ```
   pip install -r requirements.txt
   ```

2. Attendi il completamento dell'installazione
   - Questo processo potrebbe richiedere alcuni minuti

### 4.4. Verifica dell'Installazione

1. Esegui lo script di verifica:
   ```
   python setup_check.py
   ```

2. Verifica che tutte le dipendenze siano installate correttamente
   - Lo script dovrebbe mostrare "OK" per tutte le dipendenze
   - Dovrebbe rilevare la GPU NVIDIA e mostrare le informazioni relative

## 5. Risoluzione dei Problemi

### 5.1. Problemi con Python

- **Problema**: `python` non è riconosciuto come comando
  - **Soluzione**: Verifica che Python sia stato aggiunto al PATH di sistema
  - **Alternativa**: Utilizza il percorso completo, es. `C:\Python39\python.exe`

- **Problema**: Errore durante la creazione dell'ambiente virtuale
  - **Soluzione**: Installa il pacchetto `virtualenv`: `pip install virtualenv`
  - **Alternativa**: Crea l'ambiente virtuale con `virtualenv`: `virtualenv venv`

### 5.2. Problemi con CUDA

- **Problema**: `nvcc` non è riconosciuto come comando
  - **Soluzione**: Verifica che CUDA sia stato installato correttamente
  - **Alternativa**: Utilizza il percorso completo, es. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin\nvcc.exe`

- **Problema**: "CUDA driver version is insufficient for CUDA runtime version"
  - **Soluzione**: Aggiorna i driver NVIDIA alla versione più recente

### 5.3. Problemi con le Dipendenze

- **Problema**: Errore durante l'installazione delle dipendenze
  - **Soluzione**: Installa le dipendenze una alla volta: `pip install numpy`, `pip install torch`, ecc.
  - **Alternativa**: Utilizza il flag `--no-cache-dir`: `pip install --no-cache-dir -r requirements.txt`

- **Problema**: "Microsoft Visual C++ 14.0 or greater is required"
  - **Soluzione**: Installa [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## 6. Avvio dell'Applicazione

### 6.1. Interfaccia Grafica

1. Assicurati che l'ambiente virtuale sia attivato:
   ```
   venv\Scripts\activate
   ```

2. Avvia l'applicazione:
   ```
   python main.py
   ```

3. L'interfaccia grafica dovrebbe aprirsi automaticamente

### 6.2. Modalità Console

1. Assicurati che l'ambiente virtuale sia attivato:
   ```
   venv\Scripts\activate
   ```

2. Avvia l'applicazione in modalità console:
   ```
   python console_app.py
   ```

3. Segui le istruzioni a schermo per utilizzare l'applicazione

## 7. Aggiornamenti

Per aggiornare AI Parlante all'ultima versione:

1. Scarica l'ultima versione dal sito ufficiale

2. Estrai il file ZIP in una nuova directory

3. Copia la directory `voice_models` dalla vecchia installazione alla nuova per mantenere i modelli addestrati

4. Segui le istruzioni di installazione a partire dal punto 4.2

