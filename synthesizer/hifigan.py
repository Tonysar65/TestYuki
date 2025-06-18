import os
import torch
import requests
from tqdm import tqdm

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'hifigan')
MODEL_PATH = os.path.join(MODEL_DIR, 'generator_v1.pt')
# Usa il link corretto HuggingFace
URL = "https://huggingface.co/lj1995/HiFi-GAN/resolve/main/generator_v1.pt"

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    temp = MODEL_PATH + ".part"

    if os.path.exists(MODEL_PATH):
        print("Modello HiFi-GAN già presente.")
        return

    print("Scaricamento modello HiFi-GAN...")
    r = requests.get(URL, stream=True)
    total = int(r.headers.get('content-length', 0))

    with open(temp, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))

    if os.path.getsize(temp) < 10_000_000:
        os.remove(temp)
        raise RuntimeError("Download file HiFi-GAN corrotto o incompleto.")

    os.rename(temp, MODEL_PATH)
    print("Modello scaricato correttamente.")

def load_hifigan_model():
    download_model()
    model = torch.load(MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model
