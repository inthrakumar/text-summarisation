from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Corrected model import
import re

# Ensure TF_IDF is correctly imported
from example import TF_IDF  # Make sure this file exists and is accessible

def tokenize(text):
    return re.findall(r'\w+', text.lower())

# Example text
text = """
TextRank is an unsupervised graph-based ranking algorithm for NLP tasks. It is similar to PageRank, but it applies to text data...
"""  # Truncated for brevity

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('T5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('T5-base', return_dict=True)  # Changed model class

# Generate inputs and output
inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=100, truncation=True)
output = model.generate(inputs, min_length=80, max_length=100)
summary = tokenizer.decode(output[0])

# Define ROUGE n-gram calculation function
def rouge_n(reference, generated, n):
    reference_ngrams = Counter(zip(*[reference[i:i+n] for i in range(len(reference) - n + 1)]))
    generated_ngrams = Counter(zip(*[generated[i:i+n] for i in range(len(generated) - n + 1)]))

    overlap = sum((reference_ngrams & generated_ngrams).values())  # Common n-grams
    precision = overlap / len(generated_ngrams) if generated_ngrams else 0
    recall = overlap / len(reference_ngrams) if reference_ngrams else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_score

# Define LCS function
def lcs(reference, generated):
    m, n = len(reference), len(generated)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if reference[i-1] == generated[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]
    precision = lcs_length / len(generated) if len(generated) > 0 else 0
    recall = lcs_length / len(reference) if len(reference) > 0 else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_score

# Compute ROUGE scores
def compute_rouge(reference_summary, generated_summary):
    reference = tokenize(reference_summary)
    generated = tokenize(generated_summary)

    # ROUGE-1 (unigrams)
    rouge1_p, rouge1_r, rouge1_f = rouge_n(reference, generated, 1)

    # ROUGE-2 (bigrams)
    rouge2_p, rouge2_r, rouge2_f = rouge_n(reference, generated, 2)

    # ROUGE-L (Longest Common Subsequence)
    rougeL_p, rougeL_r, rougeL_f = lcs(reference, generated)

    return {
        'rouge1': {'precision': rouge1_p, 'recall': rouge1_r, 'fscore': rouge1_f},
        'rouge2': {'precision': rouge2_p, 'recall': rouge2_r, 'fscore': rouge2_f},
        'rougeL': {'precision': rougeL_p, 'recall': rougeL_r, 'fscore': rougeL_f}
    }

# Make sure TF_IDF function is properly imported and accessible
rouge_scores = compute_rouge(summary, TF_IDF(text, 3, 2))  # Ensure TF_IDF is implemented and correct

print(rouge_scores)
