# Guida Rapida - AI Parlante

## Installazione

### Requisiti
- Windows 10 (64 bit)
- Python 3.9
- NVIDIA GeForce GTX 1060 3GB o superiore
- CUDA Toolkit 11.3 o compatibile
- cuDNN compatibile con CUDA Toolkit

### Passi per l'Installazione

1. **Installazione di Python 3.9**
   ```
   # Scarica Python 3.9 dal sito ufficiale
   # Assicurati di selezionare "Add Python 3.9 to PATH" durante l'installazione
   ```

2. **Installazione di CUDA Toolkit e cuDNN**
   ```
   # Scarica e installa NVIDIA CUDA Toolkit 11.3
   # Scarica cuDNN compatibile con CUDA 11.3
   # Estrai i file cuDNN e copia i contenuti nelle rispettive directory CUDA
   ```

3. **Installazione di AI Parlante**
   ```
   # Estrai il file ZIP in una directory a tua scelta
   cd percorso/ai_parlante
   
   # Crea un ambiente virtuale
   python -m venv venv
   
   # Attiva l'ambiente virtuale
   venv\Scripts\activate
   
   # Installa le dipendenze
   pip install -r requirements.txt
   ```

4. **Verifica dell'Installazione**
   ```
   # Assicurati che l'ambiente virtuale sia attivato
   python setup_check.py
   ```

## Utilizzo Rapido

### Avvio dell'Applicazione

```
# Assicurati che l'ambiente virtuale sia attivato
python main.py
```

### Clonazione Vocale

1. Vai alla scheda "Clonazione Vocale"
2. Clicca su "Sfoglia..." e seleziona un file audio di riferimento
3. Clicca su "Analizza Audio"
4. Inserisci un nome per il modello
5. Clicca su "Addestra Modello"
6. Attendi il completamento dell'addestramento

### Sintesi Vocale

1. Vai alla scheda "Sintesi Vocale"
2. Seleziona un modello vocale dal menu a tendina
3. Inserisci il testo da sintetizzare
4. Clicca su "Sintetizza"
5. Al termine della sintesi, clicca su "Riproduci" per ascoltare l'audio generato
6. Clicca su "Salva Audio" per salvare l'audio generato

## Comandi Console

```
# Clonazione vocale
python console_app.py clone C:\audio\riferimento.wav voce_mario

# Sintesi vocale
python console_app.py synthesize voce_mario "Ciao, sono una voce sintetica creata con AI Parlante."

# Riconoscimento vocale
python console_app.py recognize C:\audio\registrazione.wav

# Elenco modelli
python console_app.py list models

# Riproduzione audio
python console_app.py play C:\audio\sintesi.wav
```

## Risoluzione dei Problemi

### Errori CUDA
- Assicurati che i driver NVIDIA siano aggiornati
- Verifica che CUDA Toolkit e cuDNN siano installati correttamente
- Riduci la dimensione del batch o la dimensione del modello
- Chiudi altre applicazioni che utilizzano la GPU

### Problemi Audio
- Verifica che i dispositivi audio siano configurati correttamente
- Riavvia l'applicazione
- Verifica che non ci siano altre applicazioni che utilizzano i dispositivi audio

### Errori di Addestramento
- Verifica che il file audio di riferimento sia di buona qualità
- Riduci la dimensione del modello o il numero di epoche
- Prova a utilizzare un file audio più lungo o più chiaro

### Errori di Sintesi
- Verifica che il modello sia stato addestrato correttamente
- Ricarica i modelli disponibili
- Prova a utilizzare un testo più breve o più semplice

