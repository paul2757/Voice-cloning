import torch
from TTS.api import TTS

# Forțăm rularea pe procesor (CPU) pentru demo-ul de pe laptop
device = "cpu"

print(f"⏳ Descarc și încarc modelul XTTS v2 pe {device}...")
# Atenție: La prima rulare, acest pas va descărca fișierele modelului (~1.8 GB).
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Textul pe care vrem să-l citească (asigură-te că folosești diacritice)
text_de_test = "Salut! Acesta este un prim test pentru a vedea cum sună vocea mea, generată complet local, fără internet."

# Fișierul cu vocea pe care vrem să o clonăm
fisier_referinta = "referinta.wav" 
fisier_iesire = "rezultat_demo.wav"

print("⚙️ Generez audio... (pe CPU va dura câteva minute pentru o propoziție)")

# Generarea propriu-zisă
tts.tts_to_file(
    text=text_de_test,
    speaker_wav=fisier_referinta,
    language="ro",
    file_path=fisier_iesire
)

print(f"✅ Gata! Fișierul a fost salvat ca: {fisier_iesire}")