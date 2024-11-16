import spacy
from tags import nouns, adjectives, verbs, articles, propernouns
# from tokenizer import Tokenizer  

nlp = spacy.load('en_core_web_sm')

class Pos:
    def __init__(self):
        pass

    def pos_tag(self, tokens):
        # tokenizer = Tokenizer(text)
        # tokens = tokenizer.word_tokenizer()
        allowed_pos = ['ADJ', 'PROPN', 'VERB', 'NOUN']
        pos_tags = []

        for word in tokens:
            doc = nlp(word)  
            if doc[0].pos_ in allowed_pos:
                pos_tags.append(doc[0].text)
        return pos_tags

    def manual_pos_tags(self, tokens):
        noun = nouns
        adjective = adjectives
        verb = verbs
        propernoun = propernouns
        pos_tags = []

        for word in tokens:
            
            if word in noun:
                pos_tags.append((word))
            elif word in adjective:
                pos_tags.append((word))
            elif word in verb:
                pos_tags.append((word))
            elif word in propernoun:
                pos_tags.append((word))
            else:
                continue
            #     doc = nlp(word)
            #     pos_tags.append((word, doc[0].pos_))
        return pos_tags



# 