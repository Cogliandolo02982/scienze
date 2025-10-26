FROM python:3.11-slim

# Install system dependencies (tesseract + italian language pack, ffmpeg, opus)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-ita \
    libtesseract-dev \
    libssl-dev \
    libffi-dev \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY bot.py /app/bot.py

ENV PORT=8080

CMD ["python", "bot.py"]
