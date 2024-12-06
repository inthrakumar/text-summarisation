import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import sent_tokenize
# import nltk

from tokenizer import Tokenizer
import stopwords



# Download required NLTK data


def textrank_summarization(text, num_sentences=3):
    """
    Summarize the input text using the TextRank algorithm.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: The summarized text.
    """
    # Step 1: Tokenize text into sentences
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return "The text is too short for summarization."
    
    # Step 2: Compute TF-IDF matrix for sentence embeddings
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Step 3: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Step 4: Create a graph and apply TextRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # Step 5: Rank sentences by score
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Step 6: Select top-ranked sentences for summary
    summary = " ".join([ranked_sentences[i][1] for i in range(num_sentences)])
    
    return summary

# Example Usage
text = """I am already far north of London, and as I walk in the streets of
Petersburgh, I feel a cold northern breeze play upon my cheeks, which
braces my nerves and fills me with delight.  Do you understand this
feeling?  This breeze, which has travelled from the regions towards
which I am advancing, gives me a foretaste of those icy climes.
Inspirited by this wind of promise, my daydreams become more fervent
and vivid.  I try in vain to be persuaded that the pole is the seat of
frost and desolation; it ever presents itself to my imagination as the
region of beauty and delight.  There, Margaret, the sun is for ever
visible, its broad disk just skirting the horizon and diffusing a
perpetual splendour.  There—for with your leave, my sister, I will put
some trust in preceding navigators—there snow and frost are banished;
and, sailing over a calm sea, we may be wafted to a land surpassing in
wonders and in beauty every region hitherto discovered on the habitable
globe.  Its productions and features may be without example, as the
phenomena of the heavenly bodies undoubtedly are in those undiscovered
solitudes.  What may not be expected in a country of eternal light?  I
may there discover the wondrous power which attracts the needle and may
regulate a thousand celestial observations that require only this
voyage to render their seeming eccentricities consistent for ever."""

summary = textrank_summarization(text, num_sentences=2)
print("Summary:", summary)
