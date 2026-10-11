import subprocess

model_path = "ro_RO-mihai-medium.onnx"
output_file = "rezultat_piper_sigur.wav"
text_de_test = "Salut! Acesta este un test folosind comanda nativă Piper rulată direct din Python. Diacriticele și punctuația funcționează perfect."

print("⚙️ Generez audio...")

# Construim comanda exact așa cum a funcționat în terminal
comanda = [
    "piper",
    "--model", model_path,
    "--output_file", output_file
]

# Rulăm comanda și trimitem textul automat, cu codare corectă pentru diacritice
proces = subprocess.run(
    comanda,
    input=text_de_test.encode("utf-8"),
    capture_output=True
)

if proces.returncode == 0:
    print(f"Gata! Fișierul a fost salvat cu succes: {output_file}")
else:
    print("A apărut o eroare în fundal:")
    print(proces.stderr.decode("utf-8"))