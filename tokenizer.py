import string

class Tokenizer:
    def __init__(self, text, stopwords):
        self.text = text
        self.stopwords = stopwords  


    def word_tokenizer(self):
        
        clean_text = ''.join([char for char in self.text if char not in string.punctuation])
        tokens = clean_text.split()

        
        filtered_tokens = [word for word in tokens if word.lower() not in self.stopwords]
        return filtered_tokens

    def sentence_tokenizer(self):
        sentences = []
        sentence = ""
        abbreviations = {"mr", "mrs", "dr", "etc", "e.g", "i.e", "vs", "st", "jr"}  
        end_punctuations = {".", "!", "?"}

        i = 0
        while i < len(self.text):
            char = self.text[i]
            sentence += char

            # Check for end of sentence
            if char in end_punctuations:
                if (i + 1 < len(self.text) and self.text[i + 1] == ".") or (
                    i > 0 and self.text[i - 1].isalpha() and self.text[i + 1:i + 3].lower() in abbreviations
                ):
                    pass
                else:
                    sentences.append(sentence.strip())
                    sentence = ""
            i += 1

        if sentence.strip():
            sentences.append(sentence.strip())

        return sentences
