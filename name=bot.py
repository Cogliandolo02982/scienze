import os
import asyncio
import tempfile
import logging
from io import BytesIO
import time

import discord
from discord import FFmpegPCMAudio
from discord.ext import commands
from PIL import Image
import pytesseract
from gtts import gTTS
import requests
from aiohttp import web

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scienze-bot")

# Environment / config
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
HF_MODEL = os.environ.get("HF_MODEL", "bigscience/bloomz-1b1")
PORT = int(os.environ.get("PORT", 8080))
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "ita")
# max characters to speak in voice (avoid very long TTS)
MAX_VOICE_CHARS = int(os.environ.get("MAX_VOICE_CHARS", 3000))

if not DISCORD_TOKEN:
    log.error("DISCORD_TOKEN non impostato. Esco.")
    raise SystemExit("DISCORD_TOKEN non impostato")

if not HUGGINGFACE_API_KEY:
    log.error("HUGGINGFACE_API_KEY non impostato. Esco.")
    raise SystemExit("HUGGINGFACE_API_KEY non impostato")

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Per evitare playback sovrapposti in una stessa guild
guild_locks = {}  # guild_id -> asyncio.Lock()

# Small webserver for Render healthcheck
async def start_webserver():
    async def handle(request):
        return web.Response(text="OK")
    app = web.Application()
    app.router.add_get('/', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    log.info(f"Health webserver avviato sulla porta {PORT}")

def call_hf_inference_sync(prompt: str, model: str = HF_MODEL, timeout: int = 60):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "return_full_text": False
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"HuggingFace error: {data.get('error')}")
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        return str(data)
    except Exception as e:
        log.exception("Errore chiamando HuggingFace Inference API")
        raise

@bot.event
async def on_ready():
    log.info(f"Bot connesso come {bot.user} (id: {bot.user.id})")

def get_guild_lock(guild_id: int):
    if guild_id not in guild_locks:
        guild_locks[guild_id] = asyncio.Lock()
    return guild_locks[guild_id]

@bot.event
async def on_message(message: discord.Message):
    # Ignora i messaggi del bot
    if message.author.bot:
        return

    # Controlla se ci sono allegati immagine
    image_attachments = []
    for att in message.attachments:
        if att.content_type and att.content_type.startswith("image"):
            image_attachments.append(att)
        else:
            if any(att.filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")):
                image_attachments.append(att)

    if not image_attachments:
        await bot.process_commands(message)
        return

    # Trova il canale vocale dell'autore
    voice_channel = None
    if isinstance(message.author, discord.Member):
        voice_channel = message.author.voice.channel if message.author.voice else None

    await message.channel.typing()

    full_extracted_text = ""
    # Scarica ed esegui OCR su ogni immagine
    for att in image_attachments:
        try:
            img_bytes = await att.read()
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            try:
                text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
            except Exception:
                text = pytesseract.image_to_string(img)
            if text:
                full_extracted_text += f"\n---\n{att.filename}:\n{text.strip()}\n"
        except Exception as e:
            log.exception("Errore durante OCR dell'allegato")
            await message.channel.send(f"Errore durante l'elaborazione dell'immagine {att.filename}: {e}")

    if not full_extracted_text.strip():
        await message.channel.send("Non sono riuscito ad estrarre testo dall'immagine. Controlla la qualità o prova a scattare una foto più nitida.")
        return

    system_prompt = (
        "Sei un assistente esperto in scienze della Terra. Riceverai del testo estratto da immagini "
        "con esercizi o domande. Rispondi in italiano fornendo la risposta dell'esercizio e, quando utile, "
        "una breve spiegazione passo-passo. Se il testo è incompleto, chiedi chiarimenti."
    )
    user_prompt = (
        "Testo estratto dall'immagine (rispondi in italiano e in modo conciso):\n\n"
        f"{full_extracted_text}\n\n"
        "Rispondi in modo chiaro e adatto a studenti."
    )

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    loop = asyncio.get_event_loop()
    try:
        answer_text = await loop.run_in_executor(None, lambda: call_hf_inference_sync(combined_prompt))
    except Exception as e:
        await message.channel.send(f"Errore durante la generazione della risposta (HuggingFace): {e}")
        return

    # Invia risultato testuale in canale
    try:
        # se troppo lungo, carica come file
        if len(answer_text) > 1900:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
                tf.write(answer_text)
                tfpath = tf.name
            await message.channel.send(f"{message.author.mention} Ho generato la risposta, la trovi nel file:", file=discord.File(tfpath))
            try:
                os.unlink(tfpath)
            except Exception:
                pass
        else:
            await message.channel.send(f"{message.author.mention} Ecco la risposta che ho trovato:\n\n{answer_text}")
    except Exception:
        log.exception("Errore inviando la risposta testuale")

    # Se l'autore è in un canale vocale, connettiti e riproduci la risposta con TTS (automatico)
    if voice_channel:
        guild_id = message.guild.id
        lock = get_guild_lock(guild_id)
        # metti in coda (lock) per evitare overlapp
        async with lock:
            # Controlla esistenza voice client per questa guild
            vc = discord.utils.get(bot.voice_clients, guild=message.guild)
            try:
                if not vc or not vc.is_connected():
                    try:
                        vc = await voice_channel.connect()
                    except discord.ClientException:
                        # già connesso altrove -> prendi client esistente
                        vc = discord.utils.get(bot.voice_clients, guild=message.guild)
            except Exception as e:
                log.exception("Errore connessione al canale vocale")
                await message.channel.send(f"Non sono riuscito a connettermi al canale vocale: {e}")
                return

            # Prepara testo per TTS (taglia se troppo lungo)
            tts_text = answer_text.strip()
            if len(tts_text) > MAX_VOICE_CHARS:
                # tronca a fine frase
                tts_text = tts_text[:MAX_VOICE_CHARS]
                # prova a tagliare fino all'ultimo punto
                last_dot = tts_text.rfind(".")
                if last_dot > 100:
                    tts_text = tts_text[:last_dot+1]

            # Genera TTS con gTTS
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tts_file = tf.name
                tts = gTTS(text=tts_text, lang="it")
                tts.save(tts_file)
            except Exception:
                log.exception("Errore durante generazione TTS")
                await message.channel.send("Errore durante la generazione del file audio.")
                try:
                    if vc and vc.is_connected():
                        await vc.disconnect()
                except Exception:
                    pass
                try:
                    if os.path.exists(tts_file):
                        os.unlink(tts_file)
                except Exception:
                    pass
                return

            # Riproduci l'audio con FFmpeg e attendi
            play_done = asyncio.Event()

            def after_play(error):
                if error:
                    log.exception("Errore durante playback audio")
                loop.call_soon_threadsafe(play_done.set)

            try:
                source = FFmpegPCMAudio(tts_file)
                vc.play(source, after=after_play)
                await play_done.wait()
            except Exception:
                log.exception("Errore durante la riproduzione audio su Discord")
                await message.channel.send("Errore durante la riproduzione audio.")
            finally:
                # pulizia file e disconnessione
                try:
                    os.unlink(tts_file)
                except Exception:
                    pass
                try:
                    if vc and vc.is_connected():
                        await vc.disconnect()
                except Exception:
                    pass
    else:
        # suggerimento testuale (automatico)
        try:
            await message.channel.send(f"{message.author.mention} Se vuoi che legga la risposta in voce, entra in un canale vocale e ripeti il caricamento dell'immagine.")
        except Exception:
            pass

# Avvio: crea task per webserver e poi avvia bot
async def main():
    await start_webserver()
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Terminato dall'utente")
