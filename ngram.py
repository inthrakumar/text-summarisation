import random
from collections import OrderedDict
import pandas as pd


df = pd.read_excel('data.xlsx')
def get_ngrams(n, text):
    words = text.split()
    words = ['<s>'] * (n - 1) + words + ['</s>']
    for i in range(len(words) - (n - 1)):
        word = words[i + n - 1]
        context = tuple(words[i:i + n - 1])
        yield (word, context)


class ngramLm:
    def __init__(self, n) -> None:
        self.n = n
        self.ngram_counts = dict()
        self.context_counts = dict()
        self.vocabulary = dict()
        self.sorted_vocab = OrderedDict()
        self.unk_words = set()
    
    def update(self, text):
        res = get_ngrams(self.n, text)
        no_words = len(text.split())
        for _ in range(no_words):
            try:
                gen = next(res)
                word = gen[0]
                context = gen[1]
                if word not in self.vocabulary:
                    self.vocabulary[word] = 1
                    self.unk_words.add(word)
                else:
                    self.vocabulary[word] += 1
                    if word in self.unk_words:
                        self.unk_words.remove(word)
                if context not in self.ngram_counts:
                    self.ngram_counts[(word, context)] = 1
                else:
                    self.ngram_counts[(word, context)] += 1
                if context not in self.context_counts:
                    self.context_counts[context] = 1
                else:
                    self.context_counts[context] += 1
            except StopIteration:
                break
                
    def word_prob(self, word, context, delta=0):
        ngram = (word, context)
        if context not in self.context_counts:
            return 1 / len(self.vocabulary)  # Probability for unseen context

        if ngram not in self.ngram_counts:
            # Handle unknown word
            if '<unk>' in self.vocabulary:
                return (self.vocabulary.get('<unk>', 0) + delta) / (self.context_counts.get(context, 0) + delta * len(self.vocabulary))
            else:
                return delta / (self.context_counts.get(context, 0) + delta * len(self.vocabulary))

        return (self.ngram_counts[ngram] + delta) / (self.context_counts[context] + delta * len(self.vocabulary))

    def random_word(self, context, delta=0):
        sorted_keys = sorted(self.vocabulary.keys())
        total_prob = 0
        word_prob = []
        for word in sorted_keys:
            total_prob += self.word_prob(word, context, delta)
            word_prob.append((word, total_prob))
        sorted_word_prob = sorted(word_prob, key=lambda x: x[1])

        r = random.random()
        for i in range(len(sorted_word_prob)):
            if sorted_word_prob[i][1] > r:
                break
        return sorted_word_prob[i - 1][0]

    def likeliest_word(self, context, delta=0):
        max_prob = -1
        likely_word = ''
        for word in self.vocabulary:
            prob = self.word_prob(word, context, delta)
            if prob >= max_prob:
                max_prob = prob
                likely_word = word
        return likely_word


def likeliest_text(model, max_length, delta=0):
    next_word = ''
    context = ('<s>',) * (model.n - 1)
    while max_length > 0 and next_word != '</s>':
        next_word = model.likeliest_word(context, delta)
        context = context[1:] + (next_word,)
        print(next_word, end=" ")
        max_length -= 1
    print()  


ngram_model = ngramLm(n=2)

# Update the model with the random text
ngram_model.update(df.iloc[0, 1])

# Generate the most likely sequence of words using the model
print("Generated text:")
likeliest_text(ngram_model, max_length=50)
