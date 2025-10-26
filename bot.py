import os
import asyncio
import tempfile
import logging
from io import BytesIO
import time
import json

import discord
from discord import FFmpegPCMAudio
from discord.ext import commands
from PIL import Image
import pytesseract
from gtts import gTTS
import requests
from aiohttp import web

# ------------------------------------------------------------
# 🔹 CONFIGURAZIONE LLM (DeepSeek o Hugging Face)
# ------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()

# DeepSeek config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = os.environ.get(
    "DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions"
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def call_deepseek_sync(system_prompt: str, user_prompt: str, timeout: int = 60):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY non impostato")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": False,
    }

    resp = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "choices" in data and data["choices"]:
        first = data["choices"][0]
        if isinstance(first, dict):
            if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                return first["message"]["content"].strip()
            if "text" in first:
                return first["text"].strip()

    if isinstance(data, dict) and "result" in data:
        return str(data["result"]).strip()

    return str(data)


# Fallback su Hugging Face
def call_hf_inference_sync(prompt: str, model: str = None, timeout: int = 60):
    headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }

    candidate_models = [
        model or os.environ.get("HF_MODEL"),
        "google/flan-t5-base",
        "google/flan-t5-small",
        "bigscience/bloom-1b1",
        "gpt2",
    ]

    last_exc = None
    for m in candidate_models:
        if not m:
            continue
        url = f"https://api-inference.huggingface.co/models/{m}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("error"):
                last_exc = RuntimeError(f"HuggingFace error for {m}: {data.get('error')}")
                continue

            if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()

            return str(data)
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(f"Nessun modello HF disponibile. Ultima eccezione: {last_exc}")


def call_llm_sync(system_prompt: str, user_prompt: str, timeout: int = 60):
    """Usa DeepSeek se LLM_PROVIDER='deepseek', altrimenti HuggingFace."""
    if LLM_PROVIDER == "deepseek":
        return call_deepseek_sync(system_prompt, user_prompt, timeout=timeout)
    combined = f"{system_prompt}\n\n{user_prompt}"
    return call_hf_inference_sync(combined, model=os.environ.get("HF_MODEL"), timeout=timeout)


# ------------------------------------------------------------
# 🔹 LOGGING E CONFIG DI BASE
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scienze-bot")

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
PORT = int(os.environ.get("PORT", 8080))
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "ita")
MAX_VOICE_CHARS = int(os.environ.get("MAX_VOICE_CHARS", 3000))

if not DISCORD_TOKEN:
    raise SystemExit("DISCORD_TOKEN non impostato")
if not HUGGINGFACE_API_KEY:
    raise SystemExit("HUGGINGFACE_API_KEY non impostato")

# ------------------------------------------------------------
# 🔹 DISCORD BOT SETUP
# ------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
guild_locks = {}  # guild_id -> asyncio.Lock()


async def start_webserver():
    async def handle(request):
        return web.Response(text="OK")

    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info(f"✅ Webserver health avviato su porta {PORT}")


def get_guild_lock(guild_id: int):
    if guild_id not in guild_locks:
        guild_locks[guild_id] = asyncio.Lock()
    return guild_locks[guild_id]


@bot.event
async def on_ready():
    log.info(f"🤖 Bot connesso come {bot.user} (id: {bot.user.id})")


# ------------------------------------------------------------
# 🔹 GESTIONE MESSAGGI CON IMMAGINI + OCR + LLM + TTS
# ------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    image_attachments = [
        att for att in message.attachments
        if (att.content_type and att.content_type.startswith("image"))
        or att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"))
    ]

    if not image_attachments:
        await bot.process_commands(message)
        return

    voice_channel = None
    if isinstance(message.author, discord.Member):
        voice_channel = message.author.voice.channel if message.author.voice else None

    await message.channel.typing()
    full_extracted_text = ""

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
            log.exception("Errore OCR")
            await message.channel.send(f"Errore durante l'elaborazione di {att.filename}: {e}")

    if not full_extracted_text.strip():
        await message.channel.send("Non sono riuscito a estrarre testo dall’immagine 😞")
        return

    system_prompt = (
        "Sei un assistente esperto in scienze della Terra. Riceverai del testo estratto da immagini "
        "con esercizi o domande. Rispondi in italiano fornendo la risposta e, se utile, una breve spiegazione passo-passo. "
        "Se il testo è incompleto, chiedi chiarimenti."
    )
    user_prompt = (
        "Testo estratto dall'immagine (rispondi in italiano e in modo conciso):\n\n"
        f"{full_extracted_text}\n\n"
        "Rispondi in modo chiaro e adatto a studenti."
    )

    loop = asyncio.get_event_loop()
    try:
        # 🔸 Usa il provider configurato (DeepSeek o HuggingFace)
        answer_text = await loop.run_in_executor(None, lambda: call_llm_sync(system_prompt, user_prompt))
    except Exception as e:
        await message.channel.send(f"Errore durante la generazione della risposta (LLM): {e}")
        return

    # 🔸 Invia risposta testuale
    if len(answer_text) > 1900:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(answer_text)
            tfpath = tf.name
        await message.channel.send(f"{message.author.mention} Ho generato la risposta:", file=discord.File(tfpath))
        os.unlink(tfpath)
    else:
        await message.channel.send(f"{message.author.mention} Ecco la risposta che ho trovato:\n\n{answer_text}")

    # 🔸 TTS automatico se l’utente è in un canale vocale
    if voice_channel:
        guild_id = message.guild.id
        lock = get_guild_lock(guild_id)

        async with lock:
            vc = discord.utils.get(bot.voice_clients, guild=message.guild)
            try:
                if not vc or not vc.is_connected():
                    vc = await voice_channel.connect()
            except Exception as e:
                await message.channel.send(f"Errore connessione al canale vocale: {e}")
                return

            tts_text = answer_text.strip()
            if len(tts_text) > MAX_VOICE_CHARS:
                tts_text = tts_text[:MAX_VOICE_CHARS]
                last_dot = tts_text.rfind(".")
                if last_dot > 100:
                    tts_text = tts_text[: last_dot + 1]

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tts_file = tf.name
                gTTS(text=tts_text, lang="it").save(tts_file)
            except Exception:
                await message.channel.send("Errore durante la generazione del file audio.")
                return

            play_done = asyncio.Event()

            def after_play(error):
                if error:
                    log.exception("Errore durante playback audio")
                loop.call_soon_threadsafe(play_done.set)

            try:
                source = FFmpegPCMAudio(tts_file)
                vc.play(source, after=after_play)
                await play_done.wait()
            finally:
                os.unlink(tts_file)
                if vc and vc.is_connected():
                    await vc.disconnect()
    else:
        await message.channel.send(
            f"{message.author.mention} Se vuoi che legga la risposta ad alta voce, entra in un canale vocale e ricarica l’immagine. 🎧"
        )


# ------------------------------------------------------------
# 🔹 AVVIO DEL BOT
# ------------------------------------------------------------
async def main():
    await start_webserver()
    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Terminato dall’utente")
