# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:25:27 2026

@author: DELL
"""

import subprocess
import datetime
import os
import time
import requests
import csv

import librosa
import numpy as np
import joblib

# -------------------------------------------------
# BASE PATH
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
OUTPUT_DIR = "/media/rpi/soundscape"
CSV_LOG = os.path.join(OUTPUT_DIR, "sound_log.csv")

MODEL_PATH = os.path.join(BASE_DIR, "sound_classifier.pkl")

NGROK_URL = "http://212b6113c0a1.ngrok-free.app/upload"

GAIN_DB = 20
RECORD_INTERVAL = 10  # seconds
NORMALIZE = True

CLASS_NAMES = ["human", "anthropogenic", "animal"]

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("[INFO] Loading sound classification model...")
model = joblib.load(MODEL_PATH)
print("[OK] Model loaded successfully")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def check_internet(url="https://www.google.com", timeout=3):
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.RequestException:
        return False


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)))


def classify_audio(wav_path):
    features = extract_features(wav_path)
    probs = model.predict_proba([features])[0]
    class_id = np.argmax(probs)
    return CLASS_NAMES[class_id], float(probs[class_id])


def log_to_csv(timestamp, audio_file, label, confidence):
    os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
    file_exists = os.path.isfile(CSV_LOG)

    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "audio_file", "class", "confidence"])
        writer.writerow([timestamp, audio_file, label, round(confidence, 3)])


def record_audio(duration=10, output_dir=OUTPUT_DIR, gain_db=20, normalize=True):
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_file = os.path.join(output_dir, f"{ts}.wav")
    mp3_file = os.path.join(output_dir, f"{ts}.mp3")

    effects = ["gain", str(gain_db)]
    if normalize:
        effects.append("norm")

    record_cmd = [
        "sox", "-t", "alsa", "-D",
        "plughw:3,0",
        "-c", "1", "-b", "16", "-r", "40000",
        wav_file,
        "trim", "0", str(duration)
    ] + effects

    try:
        print(f"[INFO] Recording {duration}s audio...")
        subprocess.run(record_cmd, check=True)

        # Convert WAV → MP3
        subprocess.run(["sox", wav_file, mp3_file], check=True)

        return wav_file, mp3_file

    except subprocess.CalledProcessError as e:
        print("[ERROR] Recording failed:", e)
        return None, None


def upload_file(mp3_path, server_url=NGROK_URL):
    if not os.path.exists(mp3_path):
        return False

    try:
        with open(mp3_path, "rb") as f:
            files = {"file": (os.path.basename(mp3_path), f, "audio/mpeg")}
            r = requests.post(server_url, files=files, timeout=30)
        return r.status_code == 200
    except requests.RequestException:
        return False


def wait_for_audio_device(timeout=60):
    print("[INFO] Waiting for USB microphone...")
    for _ in range(timeout):
        try:
            r = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            if "card 3" in r.stdout:
                print("[OK] Audio device detected")
                return True
        except:
            pass
        time.sleep(1)

    print("[ERROR] Audio device not detected")
    return False


def continuous_record_and_process():
    print("[INFO] Continuous sound monitoring started\n")

    try:
        while True:
            wav_file, mp3_file = record_audio(
                duration=RECORD_INTERVAL,
                output_dir=OUTPUT_DIR,
                gain_db=GAIN_DB,
                normalize=NORMALIZE
            )

            if wav_file and mp3_file:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                try:
                    label, confidence = classify_audio(wav_file)
                    print(f"[CLASS] {label} ({confidence:.2f})")

                    log_to_csv(
                        timestamp,
                        os.path.basename(mp3_file),
                        label,
                        confidence
                    )

                except Exception as e:
                    print("[ERROR] Classification failed:", e)

                # 🔴 DELETE WAV FILE (IMPORTANT CHANGE)
                try:
                    os.remove(wav_file)
                except:
                    pass

                # Upload ONLY MP3
                if check_internet():
                    upload_file(mp3_file)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    wait_for_audio_device()
    continuous_record_and_process()
