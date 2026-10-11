import re
import spacy
import json
import argparse
import os
from num2words import num2words

class AudiobookTextProcessor:
    def __init__(self, max_chunk_length=255):
        """
        Inițializează procesorul de text.
        :param max_chunk_length: Lungimea maximă a unui fragment.
        """
        self.max_chunk_length = max_chunk_length
        try:
            self.nlp = spacy.load("ro_core_news_sm")
        except OSError:
            raise OSError("Modelul spaCy ro_core_news_sm nu a fost găsit. Rulează: python -m spacy download ro_core_news_sm")

        self.abbreviations = {
            r'\bdl\.': 'domnul',
            r'\bd-na\b': 'doamna',
            r'\bd-nei\b': 'doamnei',
            r'\bd-lui\b': 'domnului',
            r'\bstr\.': 'strada',
            r'\bjud\.': 'județul',
            r'\betc\.': 'și celelalte',
            r'\bnr\.': 'numărul',
            r'\bap\.': 'apartamentul',
            r'\bpag\.': 'pagina',
            r'\bdr\.': 'doctorul'
        }

        self.verb_pattern = r'\b(a\s+)?(spus|zis|răspuns|adăugat|strigat|șoptit|întrebat|murmurat|zâmbit|oftat|exclamat|râs|continuat|spuse|zise|răspunse|adăugă|strigă|șopti|întrebă|murmură|zâmbi|oftă|exclamă|râse|continuă)\b'

    def normalize_text(self, text):
        text = text.replace('ş', 'ș').replace('Ş', 'Ș')
        text = text.replace('ţ', 'ț').replace('Ţ', 'Ț')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def expand_abbreviations(self, text):
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
        return text

    def expand_numbers(self, text):
        def replace_num(match):
            num_str = match.group()
            clean_num = num_str.replace(',', '.')
            try:
                words = num2words(float(clean_num), lang='ro')
                return words
            except Exception:
                return num_str

        text = re.sub(r'\b\d+(?:[.,]\d+)?\b', replace_num, text)
        return text

    def optimize_punctuation(self, text):
        text = re.sub(r'[„”«»\'`]', '"', text)
        text = re.sub(r'[\[\{]', '(', text)
        text = re.sub(r'[\]\}]', ')', text)
        text = re.sub(r'([.!?])\1+', r'\1', text) 
        text = re.sub(r'[?!]+[?!]', lambda m: m.group()[0], text)
        return text

    def extract_dialogue(self, text):
        lines = text.replace('„', '"').split('\n')
        segments = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('-') or line.startswith('—'):
                match = re.match(fr'^[—\-]\s*(.*?)([,!?])\s+({self.verb_pattern}.*)$', line, re.IGNORECASE)
                if match:
                    dialogue_text = match.group(1).strip() + match.group(2)
                    narrator_text = match.group(3).strip()
                    segments.append({"voce": "dialog", "text": dialogue_text})
                    segments.append({"voce": "narator", "text": narrator_text})
                else:
                    clean_dialogue = re.sub(r'^[—\-]\s*', '', line)
                    segments.append({"voce": "dialog", "text": clean_dialogue})
            else:
                segments.append({"voce": "narator", "text": line})
                
        return segments

    def safe_chunking(self, text):
        doc = self.nlp(text)
        chunks = []
        current_chunk = ""

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            if len(sent_text) > self.max_chunk_length:
                sub_parts = re.split(r'(?<=[,;:])\s+', sent_text)
                for part in sub_parts:
                    if len(current_chunk) + len(part) + 1 <= self.max_chunk_length:
                        current_chunk += " " + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
            else:
                if len(current_chunk) + len(sent_text) + 1 <= self.max_chunk_length:
                    current_chunk += " " + sent_text if current_chunk else sent_text
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent_text

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process(self, raw_text):
        """Aplică tot pipeline-ul pe un text brut."""
        text = re.sub(r'([.!?])\s+(?=[—\-])', r'\1\n', raw_text)
        text = self.normalize_text(text)
        text = self.expand_abbreviations(text)
        text = self.expand_numbers(text)
        text = self.optimize_punctuation(text)
        
        text = text.replace(' - ', '\n- ').replace(' — ', '\n— ')
        segments = self.extract_dialogue(text)
        
        final_pipeline = []
        chunk_id = 1 # Inițializăm un ID unic pentru fiecare fragment
        
        for segment in segments:
            chunks = self.safe_chunking(segment['text'])
            for chunk in chunks:
                if chunk.strip():
                    final_pipeline.append({
                        "id": chunk_id,
                        "voce": segment["voce"],
                        "text": chunk.strip()
                    })
                    chunk_id += 1
                    
        return final_pipeline

    def process_file(self, input_filepath, output_filepath):
        """Citește un fișier .txt, îl procesează și îl salvează ca .json"""
        if not os.path.exists(input_filepath):
            print(f"Eroare: Fișierul '{input_filepath}' nu a putut fi găsit!")
            return

        print(f"Citesc textul din {input_filepath}...")
        with open(input_filepath, 'r', encoding='utf-8') as file:
            raw_text = file.read()

        print("Procesez textul (diacritice, numere, diarizare, segmentare)...")
        json_data = self.process(raw_text)

        print(f"Salvez rezultatul în {output_filepath}...")
        # ensure_ascii=False previne transformarea caracterelor cu diacritice în coduri unicode precum \u015f
        with open(output_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
        
        print(f"Gata! Au fost generate {len(json_data)} fragmente audio pregătite pentru TTS.")

# ==========================================
# PUNCTUL DE INTRARE ÎN SCRIPT
# ==========================================
if __name__ == "__main__":
    # Configurăm argumentele din terminal
    parser = argparse.ArgumentParser(description="Procesează o carte (.txt) pentru TTS & RVC.")
    parser.add_argument("input_file", help="Calea către fișierul de intrare (.txt)")
    parser.add_argument("output_file", help="Calea către fișierul de ieșire (.json)")
    parser.add_argument("--chunk_size", type=int, default=220, help="Limita maximă de caractere per fragment (implicit: 220)")
    
    args = parser.parse_args()

    processor = AudiobookTextProcessor(max_chunk_length=args.chunk_size)
    processor.process_file(args.input_file, args.output_file)