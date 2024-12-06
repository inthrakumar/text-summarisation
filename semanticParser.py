import spacy
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

paraphrase_pipeline = pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")

def semantic_parse(sentence):

    doc = nlp(sentence)
    semantic_representation = []

    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "ROOT"):
            semantic_representation.append((token.text, token.dep_, token.head.text))
    
    return semantic_representation

def synonym_replace(word):

    synonyms = wordnet.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()
    return word  

def rephrase_sentence(sentence):
    paraphrased = paraphrase_pipeline(f"paraphrase: {sentence}", max_length=50, num_return_sequences=1)
    return paraphrased[0]["generated_text"]

def process_sentences(sentences):
    abstractive_summary = []
    for sentence in sentences:

        semantics = semantic_parse(sentence)
        print(f"Semantic Representation for '{sentence}': {semantics}")

        rephrased = rephrase_sentence(sentence)
        print(f"Rephrased Sentence: {rephrased}")
        abstractive_summary.append(rephrased)
    

    return " ".join(abstractive_summary)

text = """
Climate change is accelerating. The ice caps are melting faster than ever, which contributes to rising sea levels. 
Experts warn that if emissions aren't reduced, the planet will face irreversible damage.
"""
extractive_summary = sent_tokenize(text)[:2]  
print("Extractive Summary:", extractive_summary)
abstractive_summary = process_sentences(extractive_summary)
print("\nAbstractive Summary:", abstractive_summary)
