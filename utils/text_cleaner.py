import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha])
