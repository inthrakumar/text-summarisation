import math
import numpy as np
import re
import pandas as pd
df = pd.read_excel('data.xlsx')


# Define stopwords
stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
             'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 
             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has'}

def euclidean_distance(a):
    return math.sqrt(sum(x * x for x in a))

def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def sentence_tokenize(text):
    """Basic sentence tokenizer that splits sentences by '.', '!', or '?'."""
    return re.split(r'[.!?]', text)

def word_tokenize(sentence):
    """Tokenize sentence into words and remove punctuation."""
    if not isinstance(sentence, str):
        raise ValueError("Input must be a string")
    return re.findall(r'\b\w+\b', sentence.lower())

def remove_stopwords(tokens):
    """Remove stopwords from tokenized words."""
    return [word for word in tokens if word not in stopwords]

def get_ngrams(n, text):
    """Generate n-grams from text."""
    words = word_tokenize(text)
    words = remove_stopwords(words)
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngrams.append(' '.join(ngram))  # Join n-gram into a string
    return ngrams

def TF_IDF(text, num_sentences, n=2):
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
        
    sentences = sentence_tokenize(text)

    # Generate n-grams for all sentences and filter out stopwords
    ngram_counts = {}
    words = set()

    for sentence in sentences:
        ngrams = get_ngrams(n, sentence)
        
        for ngram in ngrams:
            words.add(ngram)  # Add n-gram to the set

            if ngram not in ngram_counts:
                ngram_counts[ngram] = 0
            ngram_counts[ngram] += 1

    # Create term frequency (TF) matrix
    td = []
    for sentence in sentences:
        sent_tf = []
        total_ngrams = len(get_ngrams(n, sentence))
        
        for word in words:
            count = sentence.count(word)
            sent_tf.append(count / total_ngrams if total_ngrams > 0 else 0)
        
        td.append(sent_tf)

    # Calculate inverse document frequency (IDF)
    idf = []
    for word in words:
        count = sum(1 for sentence in sentences if word in sentence)
        idf.append(math.log(1 + len(sentences) / (1 + count)))

    # Calculate TF-IDF matrix
    tfidf_matrix = []
    for i in range(len(td)):
        row = [td[i][j] * idf[j] for j in range(len(td[0]))]
        tfidf_matrix.append(row)

    # Create similarity matrix based on TF-IDF scores of n-grams
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                similarity_matrix[i][j] = dot_product(tfidf_matrix[i], tfidf_matrix[j]) / (
                    euclidean_distance(tfidf_matrix[i]) * euclidean_distance(tfidf_matrix[j]) + 1e-10)

    # Rank sentences based on their scores
    scores = np.ones(len(sentences)) / len(sentences)
    damping_factor = 0.85
    
    for _ in range(10000):
        new_scores = (1 - damping_factor) / len(sentences) + damping_factor * similarity_matrix.T.dot(scores)
        if np.allclose(scores, new_scores, atol=1e-4):
            break
        scores = new_scores

    ranked_sentences = [(score, sentence) for score, sentence in zip(scores, sentences)]
    ranked_sentences.sort(key=lambda x: x[0], reverse=True)

    # Generate summary from top-ranked sentences
    summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    
    print(summary)

# Example usage
text = """
    Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. 
    It enables computers to read, understand, and derive meaning from human language.
    Text summarization is a crucial task in NLP where the goal is to shorten a long text into a concise and coherent summary.
    """

# print(df.iloc[0,1])
text1=df.iloc[0,1]
# print(text1)

TF_IDF(text1, num_sentences=2, n=2)
