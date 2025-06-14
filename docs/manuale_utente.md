# Manuale Utente - AI Parlante

## Introduzione

AI Parlante è un'applicazione avanzata per la clonazione e sintesi vocale che utilizza tecnologie di intelligenza artificiale per creare voci sintetiche a partire da file audio di riferimento. Grazie al supporto per le GPU NVIDIA, l'applicazione è in grado di offrire prestazioni elevate e risultati di alta qualità.

Questo manuale fornisce istruzioni dettagliate sull'installazione, la configurazione e l'utilizzo dell'applicazione AI Parlante.

## Requisiti di Sistema

### Hardware
- **CPU**: Intel Core i5 o superiore (o equivalente AMD)
- **RAM**: 8 GB o superiore
- **GPU**: NVIDIA GeForce GTX 1060 3GB o superiore
- **Spazio su disco**: 5 GB o superiore
- **Scheda audio**: Compatibile con Windows 10
- **Microfono**: Per la registrazione audio (opzionale)

### Software
- **Sistema Operativo**: Windows 10 (64 bit)
- **Python**: 3.9
- **CUDA Toolkit**: 11.3 o compatibile
- **cuDNN**: Versione compatibile con CUDA Toolkit

## Installazione

### 1. Installazione di Python 3.9

1. Scarica Python 3.9 dal [sito ufficiale](https://www.python.org/downloads/release/python-390/)
2. Durante l'installazione, assicurati di selezionare "Add Python 3.9 to PATH"
3. Completa l'installazione seguendo le istruzioni a schermo

### 2. Installazione di CUDA Toolkit e cuDNN

Per sfruttare l'accelerazione GPU NVIDIA:

1. Scarica e installa [NVIDIA CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)
2. Scarica [cuDNN compatibile con CUDA 11.3](https://developer.nvidia.com/cudnn)
3. Estrai i file cuDNN e copia i contenuti nelle rispettive directory CUDA

### 3. Installazione di AI Parlante

#### Opzione 1: Installazione da file ZIP

1. Estrai il file ZIP in una directory a tua scelta
2. Apri un prompt dei comandi nella directory estratta
3. Crea un ambiente virtuale:
   ```
   python -m venv venv
   ```
4. Attiva l'ambiente virtuale:
   ```
   venv\Scripts\activate
   ```
5. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```

#### Opzione 2: Installazione da repository Git

1. Clona il repository:
   ```
   git clone https://github.com/Tonysar65/TestYuki.git
   ```
2. Entra nella directory del progetto:
   ```
   cd ai-parlante
   ```
3. Crea un ambiente virtuale:
   ```
   python -m venv venv
   ```
4. Attiva l'ambiente virtuale:
   ```
   venv\Scripts\activate
   ```
5. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```

### 4. Verifica dell'Installazione

Per verificare che l'installazione sia avvenuta correttamente:

1. Assicurati che l'ambiente virtuale sia attivato
2. Esegui lo script di verifica:
   ```
   python setup_check.py
   ```

Lo script verificherà che tutte le dipendenze siano installate correttamente e che la GPU NVIDIA sia riconosciuta.

## Avvio dell'Applicazione

### Interfaccia Grafica

Per avviare l'applicazione con l'interfaccia grafica:

1. Assicurati che l'ambiente virtuale sia attivato
2. Esegui il comando:
   ```
   python main.py
   ```

### Modalità Console

Per avviare l'applicazione in modalità console:

1. Assicurati che l'ambiente virtuale sia attivato
2. Esegui il comando:
   ```
   python main.py --no-gui
   ```

## Utilizzo dell'Interfaccia Grafica

L'interfaccia grafica di AI Parlante è organizzata in tre schede principali:

1. **Clonazione Vocale**: Per creare un modello vocale a partire da un file audio di riferimento
2. **Sintesi Vocale**: Per sintetizzare il parlato utilizzando un modello vocale addestrato
3. **Impostazioni**: Per configurare vari parametri dell'applicazione

### Scheda Clonazione Vocale

La scheda Clonazione Vocale permette di creare un modello vocale a partire da un file audio di riferimento.

#### Caricamento del File Audio di Riferimento

1. Clicca sul pulsante "Sfoglia..." per selezionare un file audio
2. Seleziona un file audio in formato WAV, MP3, FLAC oppure OGG
3. Il file audio verrà caricato e le informazioni (durata, frequenza di campionamento) verranno visualizzate
4. Puoi riprodurre il file audio cliccando sul pulsante "Riproduci"

#### Analisi Audio

1. Clicca sul pulsante "Analizza Audio" per estrarre le caratteristiche vocali dal file audio
2. Attendi il completamento dell'analisi (la barra di progresso mostrerà l'avanzamento)

#### Addestramento del Modello

1. Inserisci un nome per il modello nel campo "Nome del modello"
2. Imposta il numero di epoche di addestramento (più epoche = migliore qualità, ma tempi più lunghi)
3. Imposta la qualità del modello utilizzando lo slider (più alta = migliore qualità, ma richiede più memoria)
4. Clicca sul pulsante "Addestra Modello" per avviare l'addestramento
5. Attendi il completamento dell'addestramento (la barra di progresso mostrerà l'avanzamento)

### Scheda Sintesi Vocale

La scheda Sintesi Vocale permette di sintetizzare il parlato utilizzando un modello vocale addestrato.

#### Selezione del Modello

1. Seleziona un modello vocale dal menu a tendina
2. Se non vedi il modello desiderato, clicca sul pulsante "Ricarica"

#### Inserimento del Testo

1. Inserisci il testo da sintetizzare nell'editor di testo
2. Puoi caricare il testo da un file cliccando sul pulsante "Carica da File"
3. Puoi pulire l'editor di testo cliccando sul pulsante "Pulisci"

#### Parametri di Sintesi

1. Imposta la velocità di sintesi utilizzando lo slider (100% = velocità normale)
2. Imposta il tono utilizzando lo slider (0 = tono normale, valori positivi = tono più alto, valori negativi = tono più basso)

#### Sintesi e Riproduzione

1. Clicca sul pulsante "Sintetizza" per avviare la sintesi vocale
2. Attendi il completamento della sintesi (la barra di progresso mostrerà l'avanzamento)
3. Al termine della sintesi, puoi riprodurre l'audio cliccando sul pulsante "Riproduci"
4. Puoi mettere in pausa la riproduzione cliccando sul pulsante "Pausa"
5. Puoi interrompere la riproduzione cliccando sul pulsante "Stop"
6. Puoi salvare l'audio generato cliccando sul pulsante "Salva Audio"

### Scheda Impostazioni

La scheda Impostazioni permette di configurare vari parametri dell'applicazione.

#### Impostazioni Generali

- **Directory di output**: Directory in cui verranno salvati i file audio generati
- **Directory dei modelli**: Directory in cui verranno salvati i modelli vocali
- **Formato di output predefinito**: Formato predefinito per i file audio generati (WAV, MP3, FLAC, OGG)
- **Frequenza di campionamento**: Frequenza di campionamento predefinita per i file audio generati

#### Impostazioni Audio

- **Dispositivo di input**: Dispositivo di input audio predefinito
- **Dispositivo di output**: Dispositivo di output audio predefinito
- **Volume di riproduzione**: Volume predefinito per la riproduzione audio

#### Impostazioni Modello

- **Dimensione del modello**: Dimensione predefinita per i modelli vocali (Piccolo, Medio, Grande)
- **Epoche predefinite**: Numero predefinito di epoche per l'addestramento dei modelli
- **Utilizza CUDA**: Abilita o disabilita l'utilizzo di CUDA per l'accelerazione GPU
- **Precisione mista**: Abilita o disabilita l'utilizzo della precisione mista (FP16) per risparmiare memoria

#### Impostazioni Avanzate

- **Modalità debug**: Abilita o disabilita la modalità debug
- **Dimensione del batch**: Dimensione del batch per l'addestramento dei modelli
- **Dimensione FFT**: Dimensione della FFT per l'analisi audio
- **Hop length**: Hop length per l'analisi audio

## Utilizzo della Modalità Console

La modalità console permette di utilizzare AI Parlante da riga di comando, senza interfaccia grafica.

### Comandi Disponibili

- `help`: Visualizza l'elenco dei comandi disponibili
- `exit`, `quit`: Esci dall'applicazione
- `clone <file_audio> <nome_modello>`: Clona una voce
- `synthesize <nome_modello> <testo>`: Sintetizza il parlato
- `synthesize <nome_modello> -f <file>`: Sintetizza il parlato da un file
- `recognize <file_audio>`: Riconosce il parlato
- `list models`: Elenca i modelli disponibili
- `play <file_audio>`: Riproduce un file audio

### Esempi di Utilizzo

#### Clonazione Vocale

```
clone C:\audio\riferimento.wav voce_mario
```

#### Sintesi Vocale

```
synthesize voce_mario "Ciao, sono una voce sintetica creata con AI Parlante."
```

#### Sintesi Vocale da File

```
synthesize voce_mario -f C:\testi\discorso.txt
```

#### Riconoscimento Vocale

```
recognize C:\audio\registrazione.wav
```

#### Elenco Modelli

```
list models
```

#### Riproduzione Audio

```
play C:\audio\sintesi.wav
```

## Risoluzione dei Problemi

### Errori CUDA

- **Problema**: "CUDA non disponibile" o "CUDA error: out of memory"
- **Soluzione**:
  - Assicurati che i driver NVIDIA siano aggiornati
  - Verifica che CUDA Toolkit e cuDNN siano installati correttamente
  - Riduci la dimensione del batch o la dimensione del modello
  - Chiudi altre applicazioni che utilizzano la GPU

### Problemi Audio

- **Problema**: "Impossibile aprire il dispositivo audio" o "Errore durante la riproduzione"
- **Soluzione**:
  - Verifica che i dispositivi audio siano configurati correttamente
  - Riavvia l'applicazione
  - Verifica che non ci siano altre applicazioni che utilizzano i dispositivi audio

### Errori di Addestramento

- **Problema**: "Errore durante l'addestramento del modello" o "Perdita NaN"
- **Soluzione**:
  - Verifica che il file audio di riferimento sia di buona qualità
  - Riduci la dimensione del modello o il numero di epoche
  - Prova a utilizzare un file audio più lungo o più chiaro

### Errori di Sintesi

- **Problema**: "Errore durante la sintesi vocale" o "Modello non trovato"
- **Soluzione**:
  - Verifica che il modello sia stato addestrato correttamente
  - Ricarica i modelli disponibili
  - Prova a utilizzare un testo più breve o più semplice

## Appendice

### Formati Audio Supportati

- WAV
- MP3
- FLAC
- OGG

### Lingue Supportate

- Italiano
- Inglese
- Francese
- Spagnolo
- Tedesco
- Altre lingue (qualità variabile)

### Requisiti per i File Audio di Riferimento

- **Durata**: Almeno 10 secondi, idealmente 1-2 minuti
- **Qualità**: Alta qualità, senza rumori di fondo
- **Contenuto**: Parlato chiaro e naturale, senza musica o altri suoni
- **Formato**: WAV, MP3, FLAC o OGG
- **Campionamento**: Almeno 16 kHz, idealmente 22.05 kHz o 44.1 kHz
- **Canali**: Mono o stereo (verrà convertito in mono)

### Consigli per Ottenere Risultati Migliori

- Utilizza file audio di riferimento di alta qualità
- Addestra il modello con un numero sufficiente di epoche (almeno 100)
- Utilizza testi simili a quelli presenti nel file audio di riferimento
- Evita testi troppo lunghi o complessi
- Sperimenta con i parametri di sintesi (velocità, tono) per ottenere risultati migliori

