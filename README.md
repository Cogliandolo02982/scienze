```markdown
# Discord TTS Bot automatico per esercizi di Scienze della Terra

Questo repository contiene un bot Discord che:
- estrae testo da immagini (OCR con Tesseract),
- genera risposte in italiano usando la Hugging Face Inference API,
- invia la risposta in testo nel canale Discord,
- se l'autore è in un canale vocale, si connette automaticamente, legge la risposta (gTTS) e si disconnette.

Il progetto è pronto per il deploy su Render usando Docker.

## File principali
- bot.py — codice del bot automatico.
- Dockerfile — immagine Docker che installa Tesseract (con pacchetto italiano), ffmpeg e dipendenze di sistema.
- requirements.txt — dipendenze Python.
- render.yaml — esempio di configurazione per Render.

## Variabili d'ambiente (obbligatorie)
- DISCORD_TOKEN — token del bot Discord (obbligatorio).
- HUGGINGFACE_API_KEY — token Hugging Face (obbligatorio). Crea un account gratuito su https://huggingface.co/ → Settings → Access Tokens.

Opzionali:
- HF_MODEL — modello Hugging Face da usare (default: bigscience/bloomz-1b1).
- PORT — porta per il webserver di healthcheck (default: 8080).
- TESSERACT_LANG — lingua per pytesseract (default: "ita").
- MAX_VOICE_CHARS — numero massimo di caratteri letti in voce (default: 3000).

## Permessi e invito del bot
1. Vai su Discord Developer Portal → Your Apps → (seleziona l'app) → Bot:
   - Abilita "Message Content Intent" (privileged intent).
2. Per invitare il bot al server, usa OAuth2 → URL Generator:
   - Scopes: bot
   - Bot permissions: Send Messages, Connect, Speak, Read Message History, View Channels

## Deploy su Render (Docker)
1. Collega questo repository a Render e crea un nuovo Web Service di tipo Docker.
2. Render builderà l'immagine usando il Dockerfile presente nella root.
3. Imposta le environment variables su Render: DISCORD_TOKEN e HUGGINGFACE_API_KEY.
4. Avvia il servizio: Render effettuerà health-check su `/` per tenere attivo il servizio.

## Esecuzione locale (per test)
1. Installa le dipendenze di sistema (Debian/Ubuntu):
   sudo apt-get update && sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-ita
2. Crea e attiva un virtualenv:
   python -m venv .venv
   source .venv/bin/activate
3. Installa le dipendenze Python:
   pip install -r requirements.txt
4. Esporta le variabili d'ambiente:
   export DISCORD_TOKEN="<tuo_token>"
   export HUGGINGFACE_API_KEY="<tuo_token_hf>"
   export PORT=8080
5. Avvia il bot:
   python bot.py
6. Test: su Discord carica un'immagine contenente un esercizio; il bot risponderà in testo e, se sei connesso a un canale vocale, entrerà e leggerà la risposta.

## Note e consigli
- Hugging Face Inference API è gratuito con limiti di rate; per un uso intensivo valuta piani o modelli locali.
- gTTS è gratuito e richiede connettività: per una soluzione completamente offline puoi usare espeak-ng (voce meno naturale).
- Il Dockerfile include il pacchetto `tesseract-ocr-ita`; se esegui localmente assicurati di avere il pacchetto lingua italiano.
- Assicurati che il bot abbia i permessi necessari nel server Discord per parlare nei canali vocali.

## File utili che potresti aggiungere
- .gitignore — per ignorare file locali e virtualenv.
- .dockerignore — per velocizzare build Docker.
```
