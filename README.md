# Text Summarization

A Natural Language Processing (NLP) project for automatically summarizing large text documents using transformer-based models. The goal is to condense lengthy documents into shorter, meaningful summaries.

## Aim

To summarize text documents for faster comprehension and efficient information retrieval.

## Datasets

- **News Summary Dataset**: Used for training, containing diverse news articles to teach the model sentence structures and content.(source Kaggle).
- **Research Papers Dataset**: Used for testing, consisting of academic papers to evaluate summarization on complex, technical content.

## Approach

- **Transformer Models**: We use pretrained models such as **BART** and **T5** to generate abstractive summaries, rephrasing and condensing input text.
- **Preprocessing**: Includes tokenization, stopword removal, and text normalization for optimal performance.

## Evaluation

- **ROUGE**: Measures n-gram overlap between generated summaries and reference summaries.
- **BLEU**: Evaluates precision of n-grams in the generated summary.



